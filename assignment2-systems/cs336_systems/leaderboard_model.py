"""Leaderboard 8B / 32K 优化版 TransformerLM（与 cs336_basics.TransformerLM 权重与数学行为完全等价）。

集成优化：
1. FlashAttention2TritonOpt（大 tile、因果无掩码分段、Block Pointer / B200 TMA、两遍无原子反向）；
2. 延迟融合 LM Head（_FusedLMHead 继承 cs336_basics.layers.Linear，自动兼容 FSDP 分片）；
3. 共享单份 RoPE 缓存，避免 34 层重复保存 cos/sin；
4. 对三个轻量逐元素算子（RMSNorm、RoPE、SwiGLU 门控激活）按需启用 torch.compile 融合，冷启动编译仅数秒；
5. 支持选择性激活检查点：checkpoint_mode="ffn" 仅重算显存最贵但计算极快的 SwiGLU，绝不重算 32K Attention。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.layers import Embedding, Linear, RMSNorm
from cs336_systems.fused_cross_entropy import DeferredLMHead

_FLASH_ATTENTION_OPT: type[torch.autograd.Function] | None = None
try:
    from cs336_systems.flash_attention_triton_opt import FlashAttention2TritonOpt

    _FLASH_ATTENTION_OPT = FlashAttention2TritonOpt
except ImportError:
    pass


CheckpointMode = Literal["none", "ffn", "full"]


@dataclass
class LeaderboardConfig:
    ctx_len: int = 32768
    vocab_size: int = 151936
    d_model: int = 4096
    d_ff: int = 11008
    num_layers: int = 34
    num_heads: int = 32
    torch_dtype: torch.dtype = torch.bfloat16
    is_causal: bool = True
    batch_size: int = 2
    rope_theta: float = 10000.0
    eps: float = 1e-5


def _rmsnorm_eager(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * inv_rms * weight.to(torch.float32)).to(orig_dtype)


def _rope_eager(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)


def _swiglu_act_eager(gate: Tensor, up: Tensor) -> Tensor:
    orig_dtype = gate.dtype
    g = gate.to(torch.float32)
    return ((g * torch.sigmoid(g)).to(orig_dtype) * up)


_COMPILED_OPS: tuple[Any, Any, Any] | None = None


def _get_elementwise_ops(compile_ops: bool) -> tuple[Any, Any, Any]:
    global _COMPILED_OPS
    if not compile_ops or not torch.cuda.is_available():
        return _rmsnorm_eager, _rope_eager, _swiglu_act_eager
    if _COMPILED_OPS is None:
        # 仅编译 3 个极简无状态叶子函数，跨 34 层复用，空缓存冷启动仅需数秒。
        _COMPILED_OPS = (
            torch.compile(_rmsnorm_eager, fullgraph=True),
            torch.compile(_rope_eager, fullgraph=True),
            torch.compile(_swiglu_act_eager, fullgraph=True),
        )
    return _COMPILED_OPS


class _FastRMSNorm(RMSNorm):
    """继承 cs336_basics.layers.RMSNorm，保持 weight 名称与初始化一致，前向可走融合实现。"""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, *, compile_ops: bool = True) -> None:
        super().__init__(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self._compile_ops = compile_ops

    def forward(self, x: Tensor) -> Tensor:
        rmsnorm_fn, _, _ = _get_elementwise_ops(self._compile_ops)
        return rmsnorm_fn(x, self.weight, self.eps)


class _SharedRoPE(nn.Module):
    """全模型共享一份 cos/sin 缓存（非持久化 buffer，不改变 state_dict）。"""

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, *, compile_ops: bool = True) -> None:
        super().__init__()
        self._compile_ops = compile_ops
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = torch.arange(max_seq_len, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        _, rope_fn, _ = _get_elementwise_ops(self._compile_ops)
        if token_positions is None:
            seq_len = x.shape[-2]
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[token_positions]
            sin = self.sin_cached[token_positions]
        return rope_fn(x, cos, sin)


class _FastMultiHeadSelfAttention(nn.Module):
    """参数命名与 cs336_basics.attention.MultiHeadSelfAttention 完全一致，底层走 Triton FlashAttention-2。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: _SharedRoPE,
        is_causal: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_causal = is_causal
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        *lead_dims, seq_len, _ = x.shape
        batch_flat = 1
        for dim in lead_dims:
            batch_flat *= dim

        # (..., S, D) -> (B, H, S, d_k)
        q = self.q_proj(x).view(batch_flat, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_flat, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_flat, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        pos = None
        if token_positions is not None:
            pos = token_positions.broadcast_to((*lead_dims, seq_len)).reshape(batch_flat, 1, seq_len)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        if _FLASH_ATTENTION_OPT is not None and q.is_cuda and self.d_k >= 16 and (self.d_k & (self.d_k - 1)) == 0:
            bh = batch_flat * self.num_heads
            attn = _FLASH_ATTENTION_OPT.apply(
                q.reshape(bh, seq_len, self.d_k),
                k.reshape(bh, seq_len, self.d_k),
                v.reshape(bh, seq_len, self.d_k),
                self.is_causal,
            ).view(batch_flat, self.num_heads, seq_len, self.d_k)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)

        out = attn.transpose(1, 2).reshape(*lead_dims, seq_len, self.d_model)
        return self.output_proj(out)


class _FastSwiGLU(nn.Module):
    """参数命名与 cs336_basics.swiglu.SwiGLU 完全一致，融合 SiLU * up。"""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, *, compile_ops: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self._compile_ops = compile_ops
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        _, _, swiglu_act_fn = _get_elementwise_ops(self._compile_ops)
        return self.w2(swiglu_act_fn(self.w1(x), self.w3(x)))


class _FastTransformerBlock(nn.Module):
    """前置归一化 Transformer 块，支持仅对 FFN 子层做激活检查点（避免重算 32K Attention）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: _SharedRoPE,
        eps: float = 1e-5,
        is_causal: bool = True,
        device=None,
        dtype=None,
        *,
        compile_ops: bool = True,
        checkpoint_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint_ffn = checkpoint_ffn
        # 保持与 cs336_basics.transformer_block.TransformerBlock 相同的子模块注册顺序（ln1, ln2, attn, ffn）。
        self.ln1 = _FastRMSNorm(d_model, eps=eps, device=device, dtype=dtype, compile_ops=compile_ops)
        self.ln2 = _FastRMSNorm(d_model, eps=eps, device=device, dtype=dtype, compile_ops=compile_ops)
        self.attn = _FastMultiHeadSelfAttention(d_model, num_heads, rope, is_causal=is_causal, device=device, dtype=dtype)
        self.ffn = _FastSwiGLU(d_model, d_ff, device=device, dtype=dtype, compile_ops=compile_ops)

    def _ffn_sublayer(self, x: Tensor) -> Tensor:
        return self.ffn(self.ln2(x))

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        h = x + self.attn(self.ln1(x), token_positions)
        if self.checkpoint_ffn and self.training and torch.is_grad_enabled():
            ffn_out = checkpoint(self._ffn_sublayer, h, use_reentrant=False)
        else:
            ffn_out = self._ffn_sublayer(h)
        return h + ffn_out


class _FusedLMHead(Linear):
    """继承 cs336_basics.layers.Linear 以便 FSDP 自动分片；开启 defer_projection 时返回延迟句柄。"""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, *, defer_projection: bool = True) -> None:
        super().__init__(in_features, out_features, device=device, dtype=dtype)
        self.defer_projection = defer_projection

    def forward(self, x: Tensor) -> Any:
        if self.defer_projection:
            return DeferredLMHead(hidden=x, weight=self.weight)
        return super().forward(x)


class LeaderboardTransformerLM(nn.Module):
    """兼容 Leaderboard `BasicsTransformerLM(Config())` 接口的高性能 8B/32K 模型。"""

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        fuse_lm_head: bool = True,
        compile_ops: bool = True,
        checkpoint_mode: CheckpointMode = "ffn",
        checkpoint_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(cfg.vocab_size)
        ctx_len = getattr(cfg, "ctx_len", None)
        self.context_length = int(ctx_len if ctx_len is not None else cfg.context_length)
        self.d_model = int(cfg.d_model)
        self.num_layers = int(cfg.num_layers)
        self.num_heads = int(cfg.num_heads)
        self.d_ff = int(cfg.d_ff)
        self.is_causal = bool(getattr(cfg, "is_causal", True))
        self.rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        eps = float(getattr(cfg, "eps", 1e-5))
        param_dtype = dtype if dtype is not None else getattr(cfg, "torch_dtype", torch.bfloat16)

        self.checkpoint_mode = checkpoint_mode
        n_ckpt = self.num_layers if checkpoint_layers is None else max(0, min(self.num_layers, checkpoint_layers))
        self.checkpoint_layers = n_ckpt

        self.token_embeddings = Embedding(self.vocab_size, self.d_model, device=device, dtype=param_dtype)
        # 所有层共享同一份 RoPE 缓存，节省 33 份显存并保持 state_dict 与基准模型完全一致。
        shared_rope = _SharedRoPE(self.rope_theta, self.d_model // self.num_heads, self.context_length, device=device, compile_ops=compile_ops)
        self.layers = nn.ModuleList(
            [
                _FastTransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    rope=shared_rope,
                    eps=eps,
                    is_causal=self.is_causal,
                    device=device,
                    dtype=param_dtype,
                    compile_ops=compile_ops,
                    checkpoint_ffn=(checkpoint_mode == "ffn" and idx < n_ckpt),
                )
                for idx in range(self.num_layers)
            ]
        )
        self.ln_final = _FastRMSNorm(self.d_model, eps=eps, device=device, dtype=param_dtype, compile_ops=compile_ops)
        self.lm_head = _FusedLMHead(self.d_model, self.vocab_size, device=device, dtype=param_dtype, defer_projection=fuse_lm_head)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None, *, return_logits: bool = True) -> Tensor | DeferredLMHead:
        hidden_states = self.token_embeddings(token_ids)
        for idx, layer in enumerate(self.layers):
            if self.checkpoint_mode == "full" and idx < self.checkpoint_layers and self.training and torch.is_grad_enabled():
                hidden_states = checkpoint(layer, hidden_states, token_positions, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, token_positions)
        output = self.ln_final(hidden_states)
        result = self.lm_head(output)
        if return_logits and isinstance(result, DeferredLMHead):
            return result.materialize()
        return result

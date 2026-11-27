"""Assignment 2's optimized Transformer with configurable model dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers import Embedding, Linear, RMSNorm
from .loss import DeferredLMHead, cross_entropy

_FLASH_ATTENTION_OPT: type[torch.autograd.Function] | None = None
try:
    from .attention import FlashAttention2TritonOpt

    _FLASH_ATTENTION_OPT = FlashAttention2TritonOpt
except ImportError:
    pass


CheckpointMode = Literal["none", "ffn", "full"]


@dataclass(frozen=True, kw_only=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    rope_theta: float = 10000.0
    eps: float = 1e-5
    is_causal: bool = True

    def __post_init__(self) -> None:
        dimensions = (self.vocab_size, self.context_length, self.d_model, self.d_ff, self.num_layers, self.num_heads)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("Model dimensions and head count must be positive")
        if self.d_model % (2 * self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads with an even head size for RoPE")


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
        # Reuse the compiled kernels across layers.
        _COMPILED_OPS = (
            torch.compile(_rmsnorm_eager, fullgraph=True),
            torch.compile(_rope_eager, fullgraph=True),
            torch.compile(_swiglu_act_eager, fullgraph=True),
        )
    return _COMPILED_OPS


class _FastRMSNorm(RMSNorm):
    """Reuse the Assignment 2 RMSNorm parameter layout."""

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
    """Defer the output projection until logits or loss are requested."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, *, defer_projection: bool = True) -> None:
        super().__init__(in_features, out_features, device=device, dtype=dtype)
        self.defer_projection = defer_projection

    def forward(self, x: Tensor) -> Tensor | DeferredLMHead:
        if self.defer_projection:
            return DeferredLMHead(hidden=x, weight=self.weight)
        return super().forward(x)


class TransformerLM(nn.Module):
    """Return logits, or compute the mean token loss when targets are provided."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        fuse_lm_head: bool = True,
        loss_chunk_size: int = 2048,
        compile_ops: bool = True,
        checkpoint_mode: CheckpointMode = "ffn",
        checkpoint_layers: int | None = None,
    ) -> None:
        super().__init__()
        if loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive")
        if checkpoint_mode not in ("none", "ffn", "full"):
            raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.is_causal = config.is_causal
        self.rope_theta = config.rope_theta
        self.loss_chunk_size = loss_chunk_size

        self.checkpoint_mode = checkpoint_mode
        n_ckpt = self.num_layers if checkpoint_layers is None else max(0, min(self.num_layers, checkpoint_layers))
        self.checkpoint_layers = n_ckpt

        self.token_embeddings = Embedding(self.vocab_size, self.d_model, device=device, dtype=dtype)
        # Keep one shared, non-persistent RoPE cache across all layers.
        shared_rope = _SharedRoPE(self.rope_theta, self.d_model // self.num_heads, self.context_length, device=device, compile_ops=compile_ops)
        self.layers = nn.ModuleList(
            [
                _FastTransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    rope=shared_rope,
                    eps=config.eps,
                    is_causal=self.is_causal,
                    device=device,
                    dtype=dtype,
                    compile_ops=compile_ops,
                    checkpoint_ffn=(checkpoint_mode == "ffn" and idx < n_ckpt),
                )
                for idx in range(self.num_layers)
            ]
        )
        self.ln_final = _FastRMSNorm(self.d_model, eps=config.eps, device=device, dtype=dtype, compile_ops=compile_ops)
        self.lm_head = _FusedLMHead(self.d_model, self.vocab_size, device=device, dtype=dtype, defer_projection=fuse_lm_head)

    def forward(
        self,
        token_ids: Tensor,
        token_positions: Tensor | None = None,
        *,
        targets: Tensor | None = None,
        return_logits: bool = True,
    ) -> Tensor | DeferredLMHead:
        """When provided, targets must be pre-shifted and match the input token shape."""
        if targets is not None and targets.shape != token_ids.shape:
            raise ValueError("targets must have the same shape as token_ids")
        hidden_states = self.token_embeddings(token_ids)
        for idx, layer in enumerate(self.layers):
            if self.checkpoint_mode == "full" and idx < self.checkpoint_layers and self.training and torch.is_grad_enabled():
                hidden_states = checkpoint(layer, hidden_states, token_positions, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, token_positions)
        output = self.ln_final(hidden_states)
        result = self.lm_head(output)
        if targets is not None:
            return cross_entropy(result, targets, chunk_size=self.loss_chunk_size)
        if return_logits and isinstance(result, DeferredLMHead):
            return result.materialize()
        return result

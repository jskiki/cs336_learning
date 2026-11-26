"""融合版 LM Head + Cross-Entropy：按 token 分块前向并即时原地计算反向梯度，避免物化完整的 [B, S, V] logits。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from cs336_basics.cross_entropy import cross_entropy as _basics_cross_entropy

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _RCP_LN2_TL = tl.constexpr(1.4426950408889634)
    _LN2_TL = tl.constexpr(0.6931471805599453)

    @triton.jit
    def _fused_ce_fwd_bwd_kernel(
        logits_ptr,
        targets_ptr,
        losses_ptr,
        stride_row,
        V,
        inv_n,
        COMPUTE_GRAD: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """单行在线 logsumexp 交叉熵；若 COMPUTE_GRAD=True，则在第二遍直接把 d_logits 原地覆写回 logits_ptr。"""
        row = tl.program_id(0).to(tl.int64)
        row_ptr = logits_ptr + row * stride_row

        # 第一遍：以 2 为底的在线 softmax 规约，全程 FP32 累加。
        row_max = -float("inf")
        row_sum = 0.0
        for col_start in range(0, V, BLOCK_V):
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < V
            x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32) * _RCP_LN2_TL
            new_max = tl.maximum(row_max, tl.max(x, axis=0))
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(x - new_max), axis=0)
            row_max = new_max

        lse_log2 = row_max + tl.log2(row_sum)
        target_idx = tl.load(targets_ptr + row).to(tl.int64)
        target_valid = (target_idx >= 0) & (target_idx < V)
        target_logit = tl.load(row_ptr + target_idx, mask=target_valid, other=0).to(tl.float32)
        tl.store(losses_ptr + row, lse_log2 * _LN2_TL - target_logit)

        # 第二遍：原地写出 d_logits = (softmax(x) - one_hot(target)) / N，零额外 [C, V] 显存分配。
        if COMPUTE_GRAD:
            for col_start in range(0, V, BLOCK_V):
                cols = col_start + tl.arange(0, BLOCK_V)
                mask = cols < V
                x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32) * _RCP_LN2_TL
                grad = tl.exp2(x - lse_log2) * inv_n
                grad = tl.where(cols == target_idx, grad - inv_n, grad)
                tl.store(row_ptr + cols, grad, mask=mask)
except ImportError:
    pass


@dataclass(frozen=True)
class DeferredLMHead:
    """延迟执行的 LM Head 输出句柄，供 fused_cross_entropy 直接消费而不物化全量 logits。"""

    hidden: Tensor
    weight: Tensor

    def materialize(self) -> Tensor:
        return self.hidden @ self.weight.T


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """按 token 分块计算 linear + cross_entropy，并在前向内即时算出 d_hidden 与 d_weight。"""

    @staticmethod
    def forward(ctx, hidden: Tensor, weight: Tensor, targets: Tensor, chunk_size: int = 2048, grad_enabled: bool = True) -> Tensor:
        orig_shape = hidden.shape
        d_model = orig_shape[-1]
        vocab_size = weight.shape[0]

        hidden_2d = hidden.reshape(-1, d_model).contiguous()
        targets_1d = targets.reshape(-1).to(dtype=torch.long).contiguous()
        n_tokens = hidden_2d.shape[0]
        if n_tokens != targets_1d.shape[0]:
            raise ValueError(f"hidden token 数 ({n_tokens}) 与 targets ({targets_1d.shape[0]}) 不一致")
        if n_tokens == 0:
            raise ValueError("输入 token 数不能为空")

        # autograd.Function.forward 内部处于 no_grad 上下文，因此由外层传入 grad_enabled。
        needs_grad = grad_enabled and (hidden.requires_grad or weight.requires_grad)
        need_d_hidden = needs_grad and hidden.requires_grad
        need_d_weight = needs_grad and weight.requires_grad

        grad_hidden = torch.empty_like(hidden_2d) if need_d_hidden else None
        # 首块覆盖缓冲区，后续块直接累加，省去预先清零。
        grad_weight_fp32 = torch.empty_like(weight, dtype=torch.float32) if need_d_weight else None

        total_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        inv_n = 1.0 / float(n_tokens)
        use_triton = _HAS_TRITON and hidden.is_cuda
        if use_triton:
            torch._assert_async(((targets_1d >= 0) & (targets_1d < vocab_size)).all(), "cross_entropy target out of range")

        weight_t = weight.T
        if use_triton:
            block_v = min(4096, max(128, triton.next_power_of_2(vocab_size)))
            num_warps = 8 if block_v >= 2048 else 4

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            h_chunk = hidden_2d[start:end]
            t_chunk = targets_1d[start:end]
            c = end - start

            # cuBLAS BF16/FP32 GEMM 计算当前分块 logits: [C, V]（例如 2048 × 151936 仅占 ~622 MB）。
            logits_chunk = h_chunk @ weight_t
            if use_triton:
                losses_chunk = torch.empty((c,), device=hidden.device, dtype=torch.float32)
                with torch.cuda.device(hidden.device):
                    _fused_ce_fwd_bwd_kernel[(c,)](
                        logits_chunk,
                        t_chunk,
                        losses_chunk,
                        logits_chunk.stride(0),
                        vocab_size,
                        inv_n,
                        COMPUTE_GRAD=needs_grad,
                        BLOCK_V=block_v,
                        num_warps=num_warps,
                    )
                total_loss.add_(losses_chunk.sum())
                d_logits = logits_chunk
            else:
                logits_fp32 = logits_chunk.float()
                lse = torch.logsumexp(logits_fp32, dim=-1)
                target_logits = logits_fp32.gather(dim=-1, index=t_chunk.unsqueeze(-1)).squeeze(-1)
                total_loss.add_((lse - target_logits).sum())
                if needs_grad:
                    d_logits_fp32 = F.softmax(logits_fp32, dim=-1).mul_(inv_n)
                    d_logits_fp32[torch.arange(c, device=hidden.device), t_chunk] -= inv_n
                    d_logits = d_logits_fp32.to(logits_chunk.dtype)
                else:
                    d_logits = logits_chunk

            if grad_hidden is not None:
                grad_hidden[start:end] = d_logits @ weight
            if grad_weight_fp32 is not None:
                beta = 0 if start == 0 else 1
                if d_logits.dtype == h_chunk.dtype == torch.float32:
                    grad_weight_fp32.addmm_(d_logits.T, h_chunk, beta=beta)
                elif d_logits.is_cuda and d_logits.dtype == h_chunk.dtype and d_logits.dtype in (torch.float16, torch.bfloat16):
                    torch.addmm(grad_weight_fp32, d_logits.T, h_chunk, beta=beta, out_dtype=torch.float32, out=grad_weight_fp32)
                else:
                    chunk_grad = d_logits.T @ h_chunk
                    if beta == 0:
                        grad_weight_fp32.copy_(chunk_grad)
                    else:
                        grad_weight_fp32.add_(chunk_grad)

        loss = total_loss * inv_n
        saved_grad_hidden = grad_hidden.view(orig_shape) if grad_hidden is not None else None
        saved_grad_weight = grad_weight_fp32.to(weight.dtype) if grad_weight_fp32 is not None else None
        ctx.save_for_backward(saved_grad_hidden, saved_grad_weight)
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor | None, Tensor | None, None, None, None]:
        (grad_output,) = grad_outputs
        grad_hidden, grad_weight = ctx.saved_tensors
        # 避免调用 .item() 触发 CPU-GPU 同步；直接异步缩放（0-d 标量广播开销可忽略）。
        scale = grad_output.to(dtype=grad_hidden.dtype) if grad_hidden is not None else grad_output
        d_hidden = grad_hidden * scale if grad_hidden is not None else None
        if grad_weight is not None:
            w_scale = grad_output.to(dtype=grad_weight.dtype)
            d_weight = grad_weight * w_scale
        else:
            d_weight = None
        return d_hidden, d_weight, None, None, None


def fused_linear_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor, chunk_size: int = 2048) -> Tensor:
    """计算 mean(cross_entropy(hidden @ weight.T, targets))，不物化全量 logits。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    if hidden.device != weight.device or targets.device != hidden.device:
        raise ValueError("hidden、weight 和 targets 必须位于同一设备")
    return _FusedLinearCrossEntropy.apply(hidden, weight, targets, chunk_size, torch.is_grad_enabled())


def cross_entropy(inputs: Tensor | DeferredLMHead, targets: Tensor, chunk_size: int = 2048) -> Tensor:
    """兼容 DeferredLMHead（融合无全量 logits）与常规 logits 张量的统一交叉熵入口。"""
    if isinstance(inputs, DeferredLMHead):
        return fused_linear_cross_entropy(inputs.hidden, inputs.weight, targets, chunk_size=chunk_size)
    return _basics_cross_entropy(inputs, targets)

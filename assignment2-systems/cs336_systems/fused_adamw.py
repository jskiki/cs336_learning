"""Triton 融合版 AdamW 优化器（CUDA 走单核融合更新，CPU/无 Triton 环境自动回退原地 PyTorch 实现）。"""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _fused_adamw_kernel(
        p_ptr,
        g_ptr,
        m_ptr,
        v_ptr,
        n_elements,
        decay_factor,
        lr_t,
        beta1,
        one_minus_beta1,
        beta2,
        one_minus_beta2,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # 所有更新统一在 FP32 寄存器中完成，仅对 HBM 做一次读和一次写。
        p = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
        g = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)
        m = tl.load(m_ptr + offsets, mask=mask).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask).to(tl.float32)

        p = p * decay_factor
        m = beta1 * m + one_minus_beta1 * g
        v = beta2 * v + one_minus_beta2 * (g * g)
        p = p - lr_t * m / (tl.sqrt(v) + eps)

        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)


class FusedAdamW(torch.optim.Optimizer):
    """与 cs336_basics.optim.AdamW 数值等价的融合版 AdamW。

    在 CUDA 上使用单个 Triton kernel 一次读取 (p, g, m, v) 并写回 (p, m, v)，
    显著降低 HBM 读写带宽与临时显存分配；在 CPU 或未安装 Triton 时自动回退到原地 PyTorch 路径。
    支持通过 state_dtype 指定一阶/二阶矩的存储精度（默认与参数精度一致）。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        state_dtype: torch.dtype | None = None,
        block_size: int = 4096,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size 必须是正的 2 的幂，收到：{block_size}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state_dtype = state_dtype
        self.block_size = block_size

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self.state_dtype is None:
            return
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for saved_id, param in zip(saved_group["params"], group["params"], strict=True):
                saved = state_dict["state"].get(saved_id)
                if not saved:
                    continue
                for name in ("m", "v"):
                    # Optimizer.load_state_dict 会把矩状态转换为参数 dtype。
                    # 直接从 checkpoint 恢复配置的精度，避免先经历一次低精度舍入。
                    self.state[param][name] = saved[name].to(device=param.device, dtype=self.state_dtype, copy=True)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decay_factor = 1.0 - lr * weight_decay
            one_minus_beta1 = 1.0 - beta1
            one_minus_beta2 = 1.0 - beta2

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FusedAdamW 不支持稀疏梯度")

                state = self.state[p]
                if not state:
                    moment_dtype = self.state_dtype if self.state_dtype is not None else p.dtype
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, dtype=moment_dtype)
                    state["v"] = torch.zeros_like(p, dtype=moment_dtype)

                t = state["t"] + 1
                state["t"] = t
                m = state["m"]
                v = state["v"]
                lr_t = lr * sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

                n_elements = p.numel()
                if n_elements == 0:
                    continue

                if _HAS_TRITON and p.is_cuda and p.is_contiguous() and m.is_contiguous() and v.is_contiguous():
                    grad_c = grad if grad.is_contiguous() else grad.contiguous()
                    block_size = self.block_size if n_elements >= self.block_size else max(256, 1 << (n_elements - 1).bit_length())
                    num_warps = 8 if block_size >= 2048 else 4
                    grid = (triton.cdiv(n_elements, block_size),)
                    with torch.cuda.device(p.device):
                        _fused_adamw_kernel[grid](
                            p,
                            grad_c,
                            m,
                            v,
                            n_elements,
                            decay_factor,
                            lr_t,
                            beta1,
                            one_minus_beta1,
                            beta2,
                            one_minus_beta2,
                            eps,
                            BLOCK_SIZE=block_size,
                            num_warps=num_warps,
                        )
                else:
                    if m.dtype == p.dtype and grad.dtype == p.dtype:
                        if weight_decay != 0.0:
                            p.mul_(decay_factor)
                        m.mul_(beta1).add_(grad, alpha=one_minus_beta1)
                        v.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2)
                        denom = v.sqrt().add_(eps)
                        p.addcdiv_(m, denom, value=-lr_t)
                    else:
                        # 混合精度（如 BF16 参数 + FP32 矩状态）CPU 回退路径，保持 FP32 累加。
                        p_fp32 = p.float()
                        g_fp32 = grad.float()
                        m_fp32 = m.float()
                        v_fp32 = v.float()
                        if weight_decay != 0.0:
                            p_fp32.mul_(decay_factor)
                        m_fp32.mul_(beta1).add_(g_fp32, alpha=one_minus_beta1)
                        v_fp32.mul_(beta2).addcmul_(g_fp32, g_fp32, value=one_minus_beta2)
                        p_fp32.addcdiv_(m_fp32, v_fp32.sqrt().add_(eps), value=-lr_t)
                        p.copy_(p_fp32.to(p.dtype))
                        m.copy_(m_fp32.to(m.dtype))
                        v.copy_(v_fp32.to(v.dtype))

        return loss

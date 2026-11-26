"""Leaderboard 对照基线：使用原版 cs336_basics.TransformerLM + 未优化原版 FlashAttention2Triton + 原版 cross_entropy + 原版 AdamW。

不修改 cs336-basics 或已有基准脚本，仅在运行时将 MultiHeadSelfAttention 的注意力计算替换为原版
cs336_systems.flash_attention_triton.FlashAttention2Triton（32×32 分块、无因果分段、无 TMA、无 exp2），
其余全部沿用原始模型结构（无 torch.compile、无融合 LM Head/CE、无 Triton FusedAdamW），
并在 2×B200 上分别测试 ddp_sharded 与 fsdp，与优化版结果进行严格对比。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from einops import rearrange
from torch.autograd.function import once_differentiable
from torch.multiprocessing.spawn import spawn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.cross_entropy import cross_entropy as basics_cross_entropy
from cs336_basics.optim import AdamW as BasicsAdamW
from cs336_basics.transformer_lm import TransformerLM as BasicsTransformerLM
from cs336_systems.ddp import OverlapDDP
from cs336_systems.flash_attention_triton import (
    FlashAttention2Triton,
    _flash_bwd_dkv,
    _flash_bwd_dq,
    _flash_bwd_row_dot,
    flash_fwd_kernel,
)
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

WORLD_SIZE = 2
_ALLOC_STATE = threading.local()
_TUNED_KERNELS: tuple[Any, Any, Any] | None = None


def _ensure_triton_allocator() -> None:
    import triton

    if hasattr(triton, "set_allocator") and not getattr(_ALLOC_STATE, "ok", False):
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        _ALLOC_STATE.ok = True


def _get_autotuned_original_kernels() -> tuple[Any, Any, Any]:
    """直接包装原版未优化内核，仅放开 32×32 tile / warps / stages 限制（与优化版同 4 组候选）。"""
    global _TUNED_KERNELS
    if _TUNED_KERNELS is None:
        import triton

        fwd_cfgs = [triton.Config({"Q_TILE_SIZE": q, "K_TILE_SIZE": k}, num_warps=w, num_stages=s) for q, k, w, s in [(128, 64, 4, 3), (128, 128, 8, 3), (128, 64, 8, 4), (64, 64, 4, 2)]]
        dq_cfgs = [triton.Config({"Q_TILE_SIZE": q, "K_TILE_SIZE": k}, num_warps=w, num_stages=s) for q, k, w, s in [(128, 64, 8, 2), (64, 64, 4, 3), (64, 64, 4, 2), (32, 32, 4, 2)]]
        dkv_cfgs = [triton.Config({"Q_TILE_SIZE": q, "K_TILE_SIZE": k}, num_warps=w, num_stages=s) for q, k, w, s in [(64, 128, 8, 2), (64, 64, 4, 3), (64, 64, 4, 2), (32, 32, 4, 2)]]
        _TUNED_KERNELS = (
            triton.autotune(configs=fwd_cfgs, key=["N_QUERIES", "N_KEYS", "D", "is_causal"])(flash_fwd_kernel),
            triton.autotune(configs=dq_cfgs, key=["N_QUERIES", "N_KEYS", "D", "is_causal"])(_flash_bwd_dq),
            triton.autotune(configs=dkv_cfgs, key=["N_QUERIES", "N_KEYS", "D", "is_causal"])(_flash_bwd_dkv),
        )
    return _TUNED_KERNELS


@once_differentiable
def _backward_once_tuned_tiles(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    import triton

    logsumexp, query, keys, values, output = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    batch_size, n_queries, d = query.shape
    n_keys = keys.shape[1]

    row_dot = torch.empty_like(logsumexp)
    grad_query = torch.empty_like(query)
    grad_keys = torch.empty_like(keys)
    grad_values = torch.empty_like(values)

    _, dq_kernel, dkv_kernel = _get_autotuned_original_kernels()
    _ensure_triton_allocator()

    row_tile = 128 if n_queries >= 128 else 32
    dq_grid = lambda meta: (triton.cdiv(n_queries, meta["Q_TILE_SIZE"]), batch_size)  # noqa: E731
    dkv_grid = lambda meta: (triton.cdiv(n_keys, meta["K_TILE_SIZE"]), batch_size)  # noqa: E731
    scale = d**-0.5

    with torch.cuda.device(query.device):
        _flash_bwd_row_dot[(triton.cdiv(n_queries, row_tile), batch_size)](output, grad_output, row_dot, n_queries, d, row_tile)
        dq_kernel[dq_grid](query, keys, values, grad_output, logsumexp, row_dot, grad_query, n_queries, n_keys, scale, d, is_causal=ctx.is_causal)
        dkv_kernel[dkv_grid](query, keys, values, grad_output, logsumexp, row_dot, grad_keys, grad_values, n_queries, n_keys, scale, d, is_causal=ctx.is_causal)

    return (grad_query, grad_keys, grad_values, None)[: len(ctx.needs_input_grad)]


class _OriginalFlashAttentionTunedTiles(torch.autograd.Function):
    """原版未优化内核 + 仅放开 tile/warp/stage autotune 的单变量消融封装。"""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        import triton

        batch_size, n_queries, d = q.shape
        n_keys = k.shape[-2]
        query, keys, values = q.contiguous(), k.contiguous(), v.contiguous()
        output = torch.empty_like(query)
        logsumexp = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

        fwd_kernel, _, _ = _get_autotuned_original_kernels()
        _ensure_triton_allocator()
        grid = lambda meta: (triton.cdiv(n_queries, meta["Q_TILE_SIZE"]), batch_size)  # noqa: E731
        with torch.cuda.device(q.device):
            fwd_kernel[grid](
                query, keys, values, output, logsumexp,
                *query.stride(), *keys.stride(), *values.stride(), *output.stride(), *logsumexp.stride(),
                n_queries, n_keys, d**-0.5, D=d, is_causal=is_causal,
            )

        ctx.save_for_backward(logsumexp, query, keys, values, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        return _backward_once_tuned_tiles(ctx, grad_output)


def _make_mha_forward(fa_cls: type[torch.autograd.Function]):
    def _mha_forward(
        self: MultiHeadSelfAttention,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """保持原版 MultiHeadSelfAttention 的投影与 RoPE 逻辑，仅替换核心注意力调用。"""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_heads = rearrange(q, "... seq_len (num_heads d_q) -> ... num_heads seq_len d_q", num_heads=self.num_heads)
        k_heads = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v_heads = rearrange(v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if self.rope is not None:
            positions = (
                torch.arange(q_heads.shape[-2], device=q_heads.device)
                if token_positions is None
                else token_positions.unsqueeze(-2)
            )
            q_heads = self.rope(q_heads, positions)
            k_heads = self.rope(k_heads, positions)

        *lead, seq_len, d_k = q_heads.shape
        bh = 1
        for dim in lead:
            bh *= dim
        mha = fa_cls.apply(
            q_heads.reshape(bh, seq_len, d_k),
            k_heads.reshape(bh, seq_len, d_k),
            v_heads.reshape(bh, seq_len, d_k),
            True,
        ).view(*lead, seq_len, d_k)

        output = rearrange(mha, "... heads seq_len d_v -> ... seq_len (heads d_v)")
        return self.output_proj(output)

    return _mha_forward


def _patch_original_flash_attention(model: BasicsTransformerLM, *, autotune_tiles: bool = False) -> None:
    fa_cls: type[torch.autograd.Function] = _OriginalFlashAttentionTunedTiles if autotune_tiles else FlashAttention2Triton
    fn = _make_mha_forward(fa_cls)
    for mod in model.modules():
        if isinstance(mod, MultiHeadSelfAttention):
            mod.forward = MethodType(fn, mod)  # type: ignore[method-assign]


def _benchmark_worker(rank: int, args: argparse.Namespace) -> None:
    cold_start = perf_counter()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    dist.init_process_group(
        "nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=WORLD_SIZE,
        device_id=device,
    )
    phase = "init"
    try:
        init_start = perf_counter()
        torch.manual_seed(args.seed)
        base_model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.ctx_len,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=10000.0,
            device=device,
            dtype=torch.bfloat16,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        _patch_original_flash_attention(base_model, autotune_tiles=args.autotune_tiles)
        param_count = sum(p.numel() for p in base_model.parameters())

        if args.parallel == "fsdp":
            model: FSDP | OverlapDDP = FSDP(base_model, compute_dtype=torch.bfloat16)
            optimizer: torch.optim.Optimizer = BasicsAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.parallel == "ddp_sharded":
            optimizer = ShardedOptimizer(base_model.parameters(), BasicsAdamW, lr=args.lr, weight_decay=args.weight_decay)
            model = OverlapDDP(base_model, gradient_owner=optimizer.parameter_owner)
        else:
            raise ValueError(f"不支持的并行模式：{args.parallel}")

        dist.barrier()
        torch.cuda.synchronize(device)
        init_seconds = perf_counter() - init_start

        local_batch = args.batch_size // WORLD_SIZE
        torch.manual_seed(args.seed + rank)
        labels = torch.randint(high=args.vocab_size, size=(local_batch, args.ctx_len), device=device)
        targets = torch.randint(high=args.vocab_size, size=(local_batch, args.ctx_len), device=device)

        def train_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            res = model(labels)
            loss = basics_cross_entropy(res, targets).sum()
            loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()

        phase = "compile_step"
        compile_start = perf_counter()
        train_step()
        torch.cuda.synchronize(device)
        dist.barrier()
        compile_step_seconds = perf_counter() - compile_start

        phase = "estimate_step"
        est_start = torch.cuda.Event(enable_timing=True)
        est_end = torch.cuda.Event(enable_timing=True)
        est_start.record()
        train_step()
        est_end.record()
        torch.cuda.synchronize(device)
        est_ms = torch.tensor([est_start.elapsed_time(est_end)], device=device, dtype=torch.float64)
        dist.all_reduce(est_ms, op=dist.ReduceOp.MAX)
        estimate_ms = max(1.0, float(est_ms.item()))

        n_warmup = args.warmup_steps if args.warmup_steps is not None else max(1, int(args.warmup_ms / estimate_ms))
        n_rep = args.rep_steps if args.rep_steps is not None else max(3, int(args.rep_ms / estimate_ms))
        if rank == 0:
            print(
                f"[基线 {args.parallel}] 初始化 {init_seconds:.2f}s，首次编译步 {compile_step_seconds:.2f}s，"
                f"估算单步 {estimate_ms:.1f}ms -> warmup={n_warmup} 步, rep={n_rep} 步",
                flush=True,
            )

        phase = "warmup"
        for _ in range(n_warmup):
            train_step()
        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)

        phase = "measurement"
        step_times_ms: list[float] = []
        for _ in range(n_rep):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            train_step()
            end_ev.record()
            torch.cuda.synchronize(device)
            step_times_ms.append(start_ev.elapsed_time(end_ev))

        timings = torch.tensor(step_times_ms, device=device, dtype=torch.float64)
        dist.reduce(timings, dst=0, op=dist.ReduceOp.MAX)

        peak_alloc_gb = torch.tensor([torch.cuda.max_memory_allocated(device) / (1024**3)], device=device, dtype=torch.float64)
        peak_resv_gb = torch.tensor([torch.cuda.max_memory_reserved(device) / (1024**3)], device=device, dtype=torch.float64)
        all_peak_alloc = [torch.zeros_like(peak_alloc_gb) for _ in range(WORLD_SIZE)]
        all_peak_resv = [torch.zeros_like(peak_resv_gb) for _ in range(WORLD_SIZE)]
        dist.all_gather(all_peak_alloc, peak_alloc_gb)
        dist.all_gather(all_peak_resv, peak_resv_gb)

        total_wall_seconds = perf_counter() - cold_start
        if rank == 0:
            samples_ms = timings.tolist()
            mean_ms = statistics.mean(samples_ms)
            median_ms = statistics.median(samples_ms)
            std_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
            best_tiles = None
            if args.autotune_tiles and _TUNED_KERNELS is not None:
                fwd_k, dq_k, dkv_k = _TUNED_KERNELS
                best_tiles = {
                    "fwd": str(fwd_k.best_config),
                    "bwd_dq": str(dq_k.best_config),
                    "bwd_dkv": str(dkv_k.best_config),
                }
            result: dict[str, Any] = {
                "status": "ok",
                "variant": "baseline_original_model_tuned_tiles" if args.autotune_tiles else "baseline_original_model_and_fa2",
                "device_name": torch.cuda.get_device_name(device),
                "world_size": WORLD_SIZE,
                "parameter_count": param_count,
                "config": {
                    "ctx_len": args.ctx_len,
                    "vocab_size": args.vocab_size,
                    "d_model": args.d_model,
                    "d_ff": args.d_ff,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "batch_size": args.batch_size,
                    "dtype": "bfloat16",
                    "parallel": args.parallel,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "autotune_tiles": args.autotune_tiles,
                    "best_tiles": best_tiles,
                    "flash_attention": (
                        "cs336_systems.flash_attention_triton (original kernels, autotuned tiles only)"
                        if args.autotune_tiles
                        else "cs336_systems.flash_attention_triton.FlashAttention2Triton (32x32, unoptimized)"
                    ),
                    "cross_entropy": "cs336_basics.cross_entropy.cross_entropy (full logits)",
                    "optimizer": "cs336_basics.optim.AdamW (unfused)",
                    "compile_ops": False,
                    "warmup_ms": args.warmup_ms,
                    "rep_ms": args.rep_ms,
                    "n_warmup": n_warmup,
                    "n_rep": n_rep,
                },
                "init_seconds": init_seconds,
                "compile_step_seconds": compile_step_seconds,
                "total_wall_seconds": total_wall_seconds,
                "step_times_ms": samples_ms,
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "std_ms": std_ms,
                "min_ms": min(samples_ms),
                "max_ms": max(samples_ms),
                "peak_allocated_gib_by_rank": [float(t.item()) for t in all_peak_alloc],
                "peak_reserved_gib_by_rank": [float(t.item()) for t in all_peak_resv],
            }
            print(
                f"\n=== 基线结果 (原模型 + 未优化 FA2 | {result['device_name']} × {WORLD_SIZE}, {args.parallel}, grad_ckpt={args.gradient_checkpointing}) ===\n"
                f"参数量: {param_count:,} | 全局 batch={args.batch_size}, ctx_len={args.ctx_len}\n"
                f"单步耗时: median={median_ms:.2f} ms ({median_ms / 1000:.3f} s) | mean={mean_ms:.2f} ± {std_ms:.2f} ms\n"
                f"各卡峰值显存 (allocated / reserved GiB): "
                f"{[round(x, 2) for x in result['peak_allocated_gib_by_rank']]} / {[round(x, 2) for x in result['peak_reserved_gib_by_rank']]}\n"
                f"总运行时间: {total_wall_seconds:.1f} s",
                flush=True,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except torch.cuda.OutOfMemoryError as err:
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_resv = torch.cuda.max_memory_reserved(device) / (1024**3)
        print(
            f"[基线 {args.parallel} rank {rank}] OOM 于阶段 {phase} | peak_alloc={peak_alloc:.2f}GiB peak_resv={peak_resv:.2f}GiB | {err}",
            flush=True,
        )
        if rank == 0 and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "status": "oom",
                        "variant": "baseline_original_model_and_fa2",
                        "parallel": args.parallel,
                        "gradient_checkpointing": args.gradient_checkpointing,
                        "oom_phase": phase,
                        "rank0_peak_allocated_gib": peak_alloc,
                        "rank0_peak_reserved_gib": peak_resv,
                        "error": str(err),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parallel", choices=("ddp_sharded", "fsdp"), default="ddp_sharded")
    parser.add_argument("--autotune-tiles", action="store_true", help="保留原版未优化内核，仅放开 32×32 分块限制做 autotune")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--ctx-len", type=int, default=32768)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--d-ff", type=int, default=11008)
    parser.add_argument("--num-layers", type=int, default=34)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    parser.add_argument("--rep-ms", type=int, default=30_000)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--rep-steps", type=int, default=None)
    parser.add_argument("--init-method", default="tcp://127.0.0.1:29571")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"基线测试需要至少 {WORLD_SIZE} 张 CUDA GPU")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    spawn(_benchmark_worker, args=(args,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()

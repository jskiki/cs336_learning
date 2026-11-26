"""Assignment 2 Leaderboard（8B / 32K / BF16 / 2×B200）完整训练步端到端基准与正确性校验。

在双卡 GPU 上直接运行：
    uv run python -m cs336_systems.benchmark_leaderboard --output results/leaderboard.json
    uv run python -m cs336_systems.benchmark_leaderboard --parallel fsdp --checkpoint-mode ffn --quick
    uv run python -m cs336_systems.benchmark_leaderboard --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from cs336_basics.cross_entropy import cross_entropy as basics_cross_entropy
from cs336_basics.optim import AdamW as BasicsAdamW
from cs336_basics.transformer_lm import TransformerLM as BasicsTransformerLM
from cs336_systems.ddp import OverlapDDP
from cs336_systems.fsdp import FSDP
from cs336_systems.fused_adamw import FusedAdamW
from cs336_systems.fused_cross_entropy import cross_entropy as fused_cross_entropy
from cs336_systems.leaderboard_model import LeaderboardConfig, LeaderboardTransformerLM
from cs336_systems.sharded_optimizer import ShardedOptimizer

WORLD_SIZE = 2


def verify_equivalence(device: torch.device, dtype: torch.dtype, compile_ops: bool) -> dict[str, float]:
    """用小尺寸配置与 cs336_basics 基准逐项比对：logits、loss、全部参数梯度以及 1 步 AdamW 更新结果。"""
    small_cfg = LeaderboardConfig(
        ctx_len=128,
        vocab_size=257,
        d_model=128,
        d_ff=384,
        num_layers=2,
        num_heads=4,
        torch_dtype=dtype,
        is_causal=True,
        batch_size=2,
    )
    torch.manual_seed(0)
    base_model = BasicsTransformerLM(
        vocab_size=small_cfg.vocab_size,
        context_length=small_cfg.ctx_len,
        d_model=small_cfg.d_model,
        num_layers=small_cfg.num_layers,
        num_heads=small_cfg.num_heads,
        d_ff=small_cfg.d_ff,
        rope_theta=small_cfg.rope_theta,
        eps=small_cfg.eps,
        device=device,
        dtype=dtype,
    )
    opt_model = LeaderboardTransformerLM(
        small_cfg,
        device=device,
        dtype=dtype,
        fuse_lm_head=True,
        compile_ops=compile_ops,
        checkpoint_mode="ffn",
    )
    opt_model.load_state_dict(base_model.state_dict(), strict=True)

    torch.manual_seed(1)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, small_cfg.batch_size, small_cfg.ctx_len), device=device)
    labels, targets = tokens[0], tokens[1]

    # 1. 全量 logits 一致性校验
    with torch.no_grad():
        base_logits = base_model(labels)
        opt_logits = opt_model(labels)
    max_logits_diff = (base_logits.float() - opt_logits.float()).abs().max().item()

    # 2. 完整前向 + 融合交叉熵 + 反向 + AdamW 更新一致性校验
    base_opt = BasicsAdamW(base_model.parameters(), lr=1e-3, weight_decay=0.01)
    opt_opt = FusedAdamW(opt_model.parameters(), lr=1e-3, weight_decay=0.01)

    base_opt.zero_grad(set_to_none=True)
    base_loss = basics_cross_entropy(base_model(labels), targets).sum()
    base_loss.backward()

    opt_opt.zero_grad(set_to_none=True)
    opt_loss = fused_cross_entropy(opt_model(labels, return_logits=False), targets, chunk_size=64).sum()
    opt_loss.backward()

    loss_diff = abs(base_loss.item() - opt_loss.item())
    opt_params = dict(opt_model.named_parameters())
    grad_diffs = []
    initial_params = {}
    grad_atol, grad_rtol = (5e-4, 0.05) if dtype == torch.bfloat16 else (1e-5, 1e-3)
    for name, p_b in base_model.named_parameters():
        g_b = p_b.grad
        g_o = opt_params[name].grad
        assert g_b is not None and g_o is not None, f"{name} 缺少梯度"
        torch.testing.assert_close(g_o, g_b, rtol=grad_rtol, atol=grad_atol, msg=lambda msg: f"{name} 梯度不一致\n{msg}")
        grad_diffs.append((g_b.float() - g_o.float()).abs().max().item())
        initial_params[name] = p_b.detach().float().clone()
    max_grad_diff = max(grad_diffs)

    base_opt.step()
    opt_opt.step()
    param_diffs = []
    update_error_sq = 0.0
    reference_update_sq = 0.0
    for name, p_b in base_model.named_parameters():
        expected = p_b.detach().float()
        actual = opt_params[name].detach().float()
        diff = actual - expected
        param_diffs.append(diff.abs().max().item())
        update_error_sq += diff.square().sum().item()
        reference_update_sq += (expected - initial_params[name]).square().sum().item()
    max_param_diff = max(param_diffs)
    assert reference_update_sq > 0, "AdamW 基准没有产生参数更新"
    relative_update_error = (update_error_sq / reference_update_sq) ** 0.5

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    assert loss_diff <= tol, f"Loss 偏差过大：{loss_diff}"
    assert max_logits_diff <= tol * 5, f"Logits 偏差过大：{max_logits_diff}"
    assert max_param_diff <= tol, f"更新后权重偏差过大：{max_param_diff}"
    update_tol = 0.25 if dtype == torch.bfloat16 else 0.05
    assert relative_update_error <= update_tol, f"AdamW 更新量相对误差过大：{relative_update_error}"
    return {
        "max_logits_diff": max_logits_diff,
        "loss_diff": loss_diff,
        "max_grad_diff": max_grad_diff,
        "max_param_diff": max_param_diff,
        "relative_update_error": relative_update_error,
    }


def _build_parallel_system(
    rank: int,
    device: torch.device,
    cfg: LeaderboardConfig,
    args: argparse.Namespace,
) -> tuple[FSDP | OverlapDDP, torch.optim.Optimizer, int]:
    torch.manual_seed(args.seed)
    base_model = LeaderboardTransformerLM(
        cfg,
        device=device,
        dtype=cfg.torch_dtype,
        fuse_lm_head=True,
        compile_ops=not args.no_compile_ops,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_layers=args.checkpoint_layers,
    )
    param_count = sum(p.numel() for p in base_model.parameters())

    if args.parallel == "fsdp":
        # FSDP 内部以 FP32 维护本 rank 的主权重分片，计算与通信均使用 BF16。
        model: FSDP | OverlapDDP = FSDP(base_model, compute_dtype=cfg.torch_dtype)
        optimizer: torch.optim.Optimizer = FusedAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.parallel == "ddp_sharded":
        # ZeRO-1（ShardedOptimizer + OverlapDDP reduce_to_owner）：省去前向/反向全模型 AllGather，
        # 2×B200 每卡仅 ~65GB 静态显存，通常比细粒度 FSDP 更快。
        state_dtype = torch.float32 if args.fp32_optimizer_states else None
        optimizer = ShardedOptimizer(
            base_model.parameters(),
            FusedAdamW,
            lr=args.lr,
            weight_decay=args.weight_decay,
            state_dtype=state_dtype,
        )
        model = OverlapDDP(base_model, gradient_owner=optimizer.parameter_owner)
    else:
        raise ValueError(f"不支持的并行模式：{args.parallel}")

    return model, optimizer, param_count


def _benchmark_worker(rank: int, args: argparse.Namespace) -> None:
    cold_start = perf_counter()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dist.init_process_group(
        "nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=WORLD_SIZE,
        device_id=device,
    )
    try:
        verify_metrics: dict[str, float] | None = None
        if rank == 0 and (args.verify or args.verify_only):
            verify_metrics = verify_equivalence(device, torch.bfloat16, compile_ops=not args.no_compile_ops)
            print(f"[正确性校验通过] {verify_metrics}", flush=True)
        dist.barrier()
        if args.verify_only:
            return

        cfg = LeaderboardConfig(
            ctx_len=args.ctx_len,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            torch_dtype=torch.bfloat16,
            is_causal=True,
            batch_size=args.batch_size,
        )

        init_start = perf_counter()
        model, optimizer, param_count = _build_parallel_system(rank, device, cfg, args)
        dist.barrier()
        torch.cuda.synchronize(device)
        init_seconds = perf_counter() - init_start

        # 按 batch 维均分样本：全局 batch_size=2 -> 每卡 local_batch=1, ctx_len=32768。
        local_batch = cfg.batch_size // WORLD_SIZE
        torch.manual_seed(args.seed + rank)
        labels = torch.randint(high=cfg.vocab_size, size=(local_batch, cfg.ctx_len), device=device)
        targets = torch.randint(high=cfg.vocab_size, size=(local_batch, cfg.ctx_len), device=device)

        def train_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            res = model(labels, return_logits=False)
            loss = fused_cross_entropy(res, targets, chunk_size=args.chunk_size).sum()
            loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()

        # 第 1 步触发 Triton autotune / torch.compile 编译，独立计时用于监控 10 分钟冷启动限制。
        compile_start = perf_counter()
        train_step()
        torch.cuda.synchronize(device)
        dist.barrier()
        compile_step_seconds = perf_counter() - compile_start

        # 第 2 步估算稳态单步耗时，并在双卡间同步统一的 warmup/repeat 步数，防止双卡步数不一致导致 NCCL 死锁。
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
                f"初始化 {init_seconds:.2f}s，首次编译步 {compile_step_seconds:.2f}s，估算单步 {estimate_ms:.1f}ms -> "
                f"warmup={n_warmup} 步 ({args.warmup_ms}ms), rep={n_rep} 步 ({args.rep_ms}ms)",
                flush=True,
            )

        for _ in range(n_warmup):
            train_step()
        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)

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
            result: dict[str, Any] = {
                "device_name": torch.cuda.get_device_name(device),
                "world_size": WORLD_SIZE,
                "parameter_count": param_count,
                "config": {
                    "ctx_len": cfg.ctx_len,
                    "vocab_size": cfg.vocab_size,
                    "d_model": cfg.d_model,
                    "d_ff": cfg.d_ff,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "batch_size": cfg.batch_size,
                    "dtype": "bfloat16",
                    "parallel": args.parallel,
                    "checkpoint_mode": args.checkpoint_mode,
                    "checkpoint_layers": args.checkpoint_layers,
                    "chunk_size": args.chunk_size,
                    "compile_ops": not args.no_compile_ops,
                    "fp32_optimizer_states": args.fp32_optimizer_states,
                    "warmup_ms": args.warmup_ms,
                    "rep_ms": args.rep_ms,
                    "n_warmup": n_warmup,
                    "n_rep": n_rep,
                },
                "verify_metrics": verify_metrics,
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
                f"\n=== Leaderboard 结果 ({result['device_name']} × {WORLD_SIZE}, {args.parallel}, ckpt={args.checkpoint_mode}) ===\n"
                f"参数量: {param_count:,} | 全局 batch={cfg.batch_size}, ctx_len={cfg.ctx_len}\n"
                f"单步耗时: median={median_ms:.2f} ms ({median_ms / 1000:.3f} s) | mean={mean_ms:.2f} ± {std_ms:.2f} ms\n"
                f"各卡峰值显存 (allocated GiB): {[round(x, 2) for x in result['peak_allocated_gib_by_rank']]}\n"
                f"空缓存总运行时间: {total_wall_seconds:.1f} s (10 分钟上限 = 600 s)",
                flush=True,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parallel", choices=("ddp_sharded", "fsdp"), default="ddp_sharded", help="双卡并行策略")
    parser.add_argument("--checkpoint-mode", choices=("none", "ffn", "full"), default="ffn", help="激活检查点策略（ffn 仅重算 SwiGLU，不重算 32K Attention）")
    parser.add_argument("--checkpoint-layers", type=int, default=None, help="仅对前 k 层启用检查点（默认全部层）")
    parser.add_argument("--chunk-size", type=int, default=2048, help="融合 LM Head + Cross-Entropy 的 token 分块大小")
    parser.add_argument("--no-compile-ops", action="store_true", help="关闭 RMSNorm/RoPE/SwiGLU 的 torch.compile")
    parser.add_argument("--fp32-optimizer-states", action="store_true", help="ddp_sharded 模式下用 FP32 保存 m/v 矩状态")
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
    parser.add_argument("--warmup-ms", type=int, default=10_000, help="预热时长（ms），默认对齐题目 10,000 ms")
    parser.add_argument("--rep-ms", type=int, default=30_000, help="正式测量时长（ms），默认对齐题目 30,000 ms")
    parser.add_argument("--warmup-steps", type=int, default=None, help="显式指定预热步数（覆盖 --warmup-ms）")
    parser.add_argument("--rep-steps", type=int, default=None, help="显式指定测量步数（覆盖 --rep-ms）")
    parser.add_argument("--quick", action="store_true", help="快速调试模式（warmup=2000ms, rep=5000ms）")
    parser.add_argument("--verify", action="store_true", help="基准前先用小模型核验与 cs336_basics 数值等价")
    parser.add_argument("--verify-only", action="store_true", help="仅做正确性核验，不跑 8B 计时")
    parser.add_argument("--init-method", default="tcp://127.0.0.1:29511")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.quick:
        args.warmup_ms = 2_000
        args.rep_ms = 5_000
    if args.batch_size < WORLD_SIZE or args.batch_size % WORLD_SIZE != 0:
        parser.error("batch_size 必须是 2 的正整数倍")
    if args.chunk_size <= 0:
        parser.error("chunk_size 必须为正数")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"Leaderboard 基准需要至少 {WORLD_SIZE} 张 CUDA GPU")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    spawn(_benchmark_worker, args=(args,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()

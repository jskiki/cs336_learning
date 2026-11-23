"""测试 XL Transformer 在单机双卡上使用不同 DDP 同步方式的训练耗时。"""

import argparse
import json
import os
import statistics
from contextlib import nullcontext
from pathlib import Path
from timeit import default_timer

import torch
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing.spawn import spawn

from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.transformer_lm import TransformerLM
from cs336_systems.benchmark import _nvtx_range, synchronize
from cs336_systems.ddp import FlatDDP, NaiveDDP, OverlapDDP


WORLD_SIZE = 2
DDP_CLASSES = {"individual": NaiveDDP, "flat": FlatDDP, "overlap": OverlapDDP}


def train_step(
    model: NaiveDDP,
    optimizer: AdamW,
    inputs: Tensor,
    targets: Tensor,
    sync_events: tuple[torch.cuda.Event, torch.cuda.Event] | None,
    *,
    profile: bool = False,
) -> tuple[float, float]:
    step_start = default_timer()
    model.zero_grad(set_to_none=True)
    with _nvtx_range("DDP forward and loss", inputs.device) if profile else nullcontext():
        loss = cross_entropy(inputs=model(inputs), targets=targets)
    with _nvtx_range("DDP backward", inputs.device) if profile else nullcontext():
        loss.backward()

    sync_start = default_timer()
    # 这里只计反向之后的同步阶段，overlap 模式已在反向中提交部分通信。
    if sync_events is not None:
        sync_events[0].record()
    with _nvtx_range("DDP gradient sync", inputs.device) if profile else nullcontext():
        model.synchronize_gradients()
    if sync_events is not None:
        sync_events[1].record()
    #兼容cpu/Gloo模式计算耗时
    sync_seconds = default_timer() - sync_start

    with _nvtx_range("DDP optimizer", inputs.device) if profile else nullcontext():
        optimizer.step()
    #等GPU跑完
    synchronize(inputs.device)
    #得到一整步的耗时，这是单卡计时，最后结果取最慢的
    step_seconds = default_timer() - step_start
    if sync_events is not None:
        sync_seconds = sync_events[0].elapsed_time(sync_events[1]) / 1000
    return step_seconds, sync_seconds


def benchmark_worker(rank: int, args: argparse.Namespace) -> None:
    use_cuda = args.device == "cuda"
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    backend = "nccl" if use_cuda else "gloo"
    torch.set_num_threads(args.cpu_threads)
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.fp32_precision = "ieee"

    dist.init_process_group(backend, rank=rank, world_size=WORLD_SIZE, device_id=device if use_cuda else None)
    try:
        torch.manual_seed(args.seed)
        ddp_class = DDP_CLASSES[args.gradient_sync]
        model = ddp_class(
            TransformerLM(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                d_model=args.d_model,
                d_ff=args.d_ff,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                rope_theta=args.rope_theta,
                device=device,
                dtype=torch.float32,
            )
        )
        optimizer = AdamW(model.parameters(), lr=args.lr)

        # 复用单卡的随机数据构造方式，再沿 batch 维给每个 rank 分一半。
        shape = (args.batch_size, args.context_length + 1)
        tokens = (
            torch.randint(args.vocab_size, shape, device=device)
            if rank == 0
            else torch.empty(shape, dtype=torch.long, device=device)
        )
        dist.broadcast(tokens, src=0)
        local_tokens = tokens.chunk(WORLD_SIZE, dim=0)[rank]
        inputs = local_tokens[:, :-1].contiguous()
        targets = local_tokens[:, 1:].contiguous()

        sync_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if use_cuda else None
        synchronize(device)
        for _ in range(args.warmup_steps):
            train_step(model, optimizer, inputs, targets, sync_events)

        dist.barrier()
        synchronize(device)
        # rank 0 触发整个进程树的采集，两个 rank 都完成后才关闭采集范围。
        with _nvtx_range("measurement", device) if args.profile and rank == 0 else nullcontext():
            if args.profile:
                dist.barrier()
                synchronize(device)
            samples = [train_step(model, optimizer, inputs, targets, sync_events, profile=args.profile) for _ in range(args.steps)]
            if args.profile:
                dist.barrier()
                synchronize(device)

        # 对同一步的两项耗时分别取最慢 rank。统计归约在正式计时之外。
        timings = torch.tensor(samples, device=device, dtype=torch.float64)
        dist.reduce(timings, dst=0, op=dist.ReduceOp.MAX)
        if rank == 0:
            step_times, sync_times = timings.T.tolist()
            step_mean = statistics.mean(step_times)
            sync_mean = statistics.mean(sync_times)
            overlap = args.gradient_sync == "overlap"
            result = {
                "config": {key: value for key, value in vars(args).items() if key != "output"} | {"dtype": "float32"},
                "world_size": WORLD_SIZE,
                "local_batch_size": inputs.size(0),
                "backend": backend,
                "device_name": torch.cuda.get_device_name(device) if use_cuda else "cpu",
                "torch_version": str(torch.__version__),
                "cuda_runtime_version": torch.version.cuda,
                "cuda_matmul_allow_tf32": False if use_cuda else None,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "timing_aggregation": "max across ranks for each metric and step; sample standard deviation",
                "step_times_seconds": step_times,
                "mean_seconds": step_mean,
                "std_seconds": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
                "gradient_sync_timing_scope": "post_backward_tail" if overlap else "full_gradient_sync",
                "gradient_sync_times_seconds": sync_times,
                "gradient_sync_mean_seconds": sync_mean,
                "gradient_sync_std_seconds": statistics.stdev(sync_times) if len(sync_times) > 1 else 0.0,
                "gradient_sync_percent": 100 * sync_mean / step_mean,
            }
            print(
                f"{result['device_name']} / {backend}，FP32，{result['parameter_count']:,} 参数，"
                f"全局 batch {args.batch_size}，每卡 batch {inputs.size(0)}，序列长度 {args.context_length}"
            )
            print(f"预热 {args.warmup_steps} 步，测量 {args.steps} 步，梯度同步模式 {args.gradient_sync}。")
            if overlap:
                sync_label = "反向后同步尾段"
                print("反向后同步尾段包含剩余通信的提交与等待，以及本地梯度平均。")
            else:
                sync_label = "梯度同步"
                print("梯度同步耗时包括 all-reduce、本地平均，以及 flat 模式的拼接和拆回。")
            print(f"| 单步总耗时 (ms) | {sync_label} (ms) | 占单步比例 |")
            print("| ---: | ---: | ---: |")
            print(
                f"| {step_mean * 1000:.3f} ± {result['std_seconds'] * 1000:.3f} "
                f"| {sync_mean * 1000:.3f} ± {result['gradient_sync_std_seconds'] * 1000:.3f} "
                f"| {result['gradient_sync_percent']:.2f}% |",
                flush=True,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CPU/Gloo 用于小尺寸本地验证")
    parser.add_argument("--gradient-sync", choices=DDP_CLASSES, default="individual", help="逐参数、展平合并或与反向重叠的梯度同步")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4, help="全局 batch size")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4, help="每个 rank 的 PyTorch CPU 线程数")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="为 Nsight Systems 标记正式测量与各训练阶段")
    parser.add_argument("--output", type=Path, help="可选 JSON 结果路径")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.batch_size % WORLD_SIZE:
        parser.error("全局 batch size 必须是 2 的正整数倍")
    if args.steps <= 0 or args.warmup_steps < 0 or args.cpu_threads <= 0:
        parser.error("测量步数和 CPU 线程数必须为正，预热步数可以为 0")
    if args.device == "cuda" and torch.cuda.device_count() < WORLD_SIZE:
        parser.error("CUDA 基准需要同一台机器上有两张可见 GPU")

    #这不是用来all_reduce的，是让rank交换连接信息的，TCPStore 传递的关键信息是一个两张卡共用的NCCL初始化ID
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    spawn(benchmark_worker, args=(args,), nprocs=WORLD_SIZE)


if __name__ == "__main__":
    main()

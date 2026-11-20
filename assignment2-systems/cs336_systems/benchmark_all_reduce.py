"""用 CPU/Gloo 或 GPU/NCCL 测量单机多进程 all-reduce 的耗时。"""

import argparse
import os
import statistics
from time import perf_counter

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn


def benchmark_worker(
    rank: int,
    world_size: int,
    sizes_mb: list[int],
    warmup_steps: int,
    steps: int,
    backend: str,
) -> None:
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    use_cuda = backend == "nccl"
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=device if use_cuda else None)
    try:
        expected_sum = world_size * (world_size + 1) // 2
        for size_mb in sizes_mb:
            # 1 MB = 1,000,000 字节，FP32 每个元素占 4 字节。
            data = torch.empty(size_mb * 1_000_000 // 4, dtype=torch.float32, device=device)
            for _ in range(warmup_steps):
                data.fill_(rank + 1)
                dist.all_reduce(data)

            times_ms: list[float] = []
            for _ in range(steps):
                # all_reduce 会原地覆盖输入。重置输入和对齐各 rank 都放在计时外。
                data.fill_(rank + 1)
                #等到其他rank准备好，尽量对齐计时起点，只测all_reduce的时间
                dist.barrier()
                if use_cuda:
                    torch.cuda.synchronize(device)
                start = perf_counter()
                dist.all_reduce(data, async_op=False)
                if use_cuda:
                    torch.cuda.synchronize(device)
                times_ms.append((perf_counter() - start) * 1000)

            if data[0].item() != expected_sum:
                raise RuntimeError(f"rank {rank} 的归约结果与预期不符")

            # 先取每轮最慢 rank 的耗时，再汇总各轮。
            times = torch.tensor(times_ms, dtype=torch.float64, device=device)
            #所有 rank 都调用这一行。op=MAX表示按对应下标取最大值，dst=0表示把结果写入 rank 0 的 times
            dist.reduce(times, dst=0, op=dist.ReduceOp.MAX)
            if rank == 0:
                latencies = times.tolist()
                print(
                    f"| {world_size} | {size_mb} | {statistics.median(latencies):.3f} " #median:中位数
                    f"| {min(latencies):.3f} | {max(latencies):.3f} |",
                    flush=True,
                )
            del data
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo", help="通信后端")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 6], help="进程数")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[1, 10, 100, 1000], help="每个 rank 的张量大小，单位 MB")
    parser.add_argument("--warmup-steps", type=int, default=5, help="每组预热次数")
    parser.add_argument("--steps", type=int, default=10, help="每组计时次数")
    args = parser.parse_args()
    if any(size <= 0 for size in args.world_sizes + args.sizes_mb) or args.warmup_steps < 0 or args.steps <= 0:
        parser.error("进程数、张量大小和计时次数必须为正数，预热次数可以为 0")
    if args.backend == "nccl":
        gpu_count = torch.cuda.device_count()
        if max(args.world_sizes) > gpu_count:
            parser.error(f"NCCL 需要每个进程一张 GPU，目前可见 {gpu_count} 张")

    backend_label = "GPU / NCCL" if args.backend == "nccl" else "CPU / Gloo"
    print(f"{backend_label}，FP32，每组预热 {args.warmup_steps} 次，计时 {args.steps} 次\n")
    print("| 进程数 | 单 rank 大小 (MB) | 中位数 (ms) | 最快 (ms) | 最慢 (ms) |")
    print("| ---: | ---: | ---: | ---: | ---: |", flush=True)
    for world_size in args.world_sizes:
        spawn(
            benchmark_worker,
            args=(world_size, args.sizes_mb, args.warmup_steps, args.steps, args.backend),
            nprocs=world_size,
        )


if __name__ == "__main__":
    main()

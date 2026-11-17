"""作业 4.1.1：对自己的 PyTorch attention 做前向、反向和显存基准测试。

在 assignment2-systems 目录运行（需要 CUDA GPU）：
    uv run python -m cs336_systems.benchmark_attention --output results/pytorch-attention.json
    uv run python -m cs336_systems.benchmark_attention --causal --output results/causal-attention.json

默认 batch=8、FP32、无 mask；遍历题目全部 20 组配置，预热 5 轮后测量 100 轮。
每轮保留训练前向的计算图，分别计时一次 forward 和一次 backward。
CPU 仅用于小尺寸流程检查，不产生显存数据：
    uv run python -m cs336_systems.benchmark_attention --device cpu --seq-lens 16 --d-models 16 --steps 2
"""

import argparse
import gc
import json
import statistics
from itertools import product
from pathlib import Path
from timeit import default_timer

import torch

from cs336_basics.attention import scaled_dot_product_attention


BATCH_SIZE = 8
MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda", help="默认 cuda；cpu 仅用于小尺寸流程检查")
    parser.add_argument("--causal", action="store_true", help="启用下三角因果 mask；mask 在计时外构造")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="可选 JSON 路径，每完成一组配置就保存一次")
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup_steps < 0:
        parser.error("steps 必须为正数，warmup-steps 不能为负数")
    if any(value <= 0 for value in args.d_models + args.seq_lens):
        parser.error("d-models 和 seq-lens 必须为正数")
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        parser.error("device 必须是 CUDA 或 CPU")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用；请在 GPU 环境运行，或使用 --device cpu 检查小尺寸流程")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_case(args: argparse.Namespace, device: torch.device, result: dict) -> None:
    """单组配置的张量仅存在于此函数中，返回或 OOM 后由调用方统一清理。"""
    torch.manual_seed(args.seed)
    result["phase"] = "inputs"
    shape = (BATCH_SIZE, result["seq_len"], result["d_model"])
    dtype = getattr(torch, args.dtype)
    q, k, v = [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
    # 直接提供输出梯度，避免额外的 loss/reduction 混入反向计时。
    grad_output = torch.randn(shape, device=device, dtype=dtype)
    mask = None
    if args.causal:
        positions = torch.arange(shape[1], device=device)
        mask = positions[:, None] >= positions[None, :]
    synchronize(device)

    with torch.enable_grad():
        for _ in range(args.warmup_steps):
            result["phase"] = "warmup_forward"
            output = scaled_dot_product_attention(q, k, v, mask)
            synchronize(device)
            result["phase"] = "warmup_backward"
            output.backward(grad_output)
            synchronize(device)
            del output
            q.grad = k.grad = v.grad = None

        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result["peak_scope"] = "measurement"

        for _ in range(args.steps):
            # 清理和同步放在计时外；每一轮都创建新的计算图，不使用 retain_graph。
            q.grad = k.grad = v.grad = None
            synchronize(device)
            baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

            result["phase"] = "forward"
            start = default_timer()
            output = scaled_dot_product_attention(q, k, v, mask)
            synchronize(device)
            result["forward_times_ms"].append((default_timer() - start) * 1000)

            # 此时 forward 已完成，backward 尚未开始，计算图仍然存活。
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device)
                result["before_backward_allocated_bytes"].append(allocated)
                result["before_backward_reserved_bytes"].append(torch.cuda.memory_reserved(device))
                result["forward_live_delta_bytes"].append(allocated - baseline)

            result["phase"] = "backward"
            start = default_timer()
            output.backward(grad_output)
            synchronize(device)
            result["backward_times_ms"].append((default_timer() - start) * 1000)
            del output
        q.grad = k.grad = v.grad = None
    result["phase"] = "complete"


def summarize(result: dict) -> None:
    for phase in ("forward", "backward"):
        samples = result[f"{phase}_times_ms"]
        result[f"{phase}_completed_steps"] = len(samples)
        result[f"{phase}_mean_ms"] = statistics.mean(samples) if samples else None
        result[f"{phase}_std_ms"] = statistics.stdev(samples) if len(samples) > 1 else None
    for field in ("before_backward_allocated_bytes", "before_backward_reserved_bytes", "forward_live_delta_bytes"):
        samples = result[field]
        result[f"{field}_max"] = max(samples) if samples else None


def table_row(result: dict) -> str:
    def formatted(value: float | None, divisor: float = 1) -> str:
        return f"{value / divisor:.3f}" if value is not None else "—"

    status = "OK" if result["status"] == "ok" else f"OOM ({result['phase']})"
    # OOM 行不把部分完成的计时作为完整的 100 轮结果；原始部分数据仍保存在 JSON。
    forward = formatted(result["forward_mean_ms"]) if result["status"] == "ok" else "—"
    backward = formatted(result["backward_mean_ms"]) if result["status"] == "ok" else "—"
    memory = formatted(result["before_backward_allocated_bytes_max"], MIB)
    delta = formatted(result["forward_live_delta_bytes_max"], MIB)
    return f"| {result['d_model']} | {result['seq_len']} | {status} | {forward} | {backward} | {memory} | {delta} |"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
    else:
        print("CPU 流程检查：耗时不是 GPU 基准结果，显存列为 —。", flush=True)

    report = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "batch_size": BATCH_SIZE,
        "attention_function": "cs336_basics.attention.scaled_dot_product_attention",
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "timing_scope": "wall clock, grad-enabled forward and backward timed separately, synchronized after each pass",
        "memory_scope": "PyTorch allocator; before-backward columns show maxima over measured forwards",
        "forward_live_delta_note": "allocated after forward minus allocated before forward; includes output and saved tensors, not an exact saved-for-backward count",
        "results": [],
    }
    print(f"device={report['device_name']} dtype={args.dtype} batch={BATCH_SIZE} causal={args.causal} warmup={args.warmup_steps} steps={args.steps}", flush=True)
    print("| d_model | seq_len | status | forward mean (ms) | backward mean (ms) | before backward (MiB) | forward live delta (MiB) |", flush=True)
    print("| --- | --- | --- | --- | --- | --- | --- |", flush=True)

    for d_model, seq_len in product(args.d_models, args.seq_lens):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        result = {
            "d_model": d_model,
            "seq_len": seq_len,
            "status": "ok",
            "phase": "inputs",
            "peak_scope": "inputs_and_warmup",
            "forward_times_ms": [],
            "backward_times_ms": [],
            "before_backward_allocated_bytes": [],
            "before_backward_reserved_bytes": [],
            "forward_live_delta_bytes": [],
        }
        try:
            measure_case(args, device, result)
        except torch.cuda.OutOfMemoryError as error:
            if device.type != "cuda":
                raise
            result["status"] = "oom"
            result["error"] = str(error)
        # 在 except 块外清理，确保异常 traceback 不再持有失败函数中的张量。
        if device.type == "cuda":
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        summarize(result)
        report["results"].append(result)
        print(table_row(result), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    oom_shapes = [(row["d_model"], row["seq_len"]) for row in report["results"] if row["status"] == "oom"]
    print(f"\nOOM 配置 (d_model, seq_len)：{oom_shapes or '无'}", flush=True)
    if args.output is not None:
        print(f"原始逐轮计时、显存数据和 OOM 信息已保存到：{args.output}", flush=True)


if __name__ == "__main__":
    main()

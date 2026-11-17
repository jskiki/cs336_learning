"""作业 4.1.1 / 4.2(a)：对自己的 eager / compiled attention 做基准测试。

在 assignment2-systems 目录运行（需要 CUDA GPU）：
    uv run python -m cs336_systems.benchmark_attention --output results/pytorch-attention.json
    uv run python -m cs336_systems.benchmark_attention --causal --output results/causal-attention.json
    uv run python -m cs336_systems.benchmark_attention --compile --baseline results/pytorch-attention.json --output results/compiled-attention.json

默认 batch=8、FP32、无 mask；遍历题目全部 20 组配置，预热 5 轮后测量 100 轮。
每轮保留训练前向的计算图，分别计时一次 forward 和一次 backward。
--compile 使用 Inductor，对每组形状独立编译；首次前向、反向编译开销由预热承担。
--baseline 读取同配置的 eager JSON，打印对比表，并在指定 output 时保存同名 .comparison.md。
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
    parser.add_argument("--compile", action="store_true", help="使用 torch.compile，编译开销排除在正式计时外")
    parser.add_argument("--baseline", type=Path, help="同配置的未编译 JSON 结果，用于生成前向/反向对比表")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="可选 JSON 路径，每完成一组配置就保存一次")
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup_steps < 0:
        parser.error("steps 必须为正数，warmup-steps 不能为负数")
    if args.compile and args.warmup_steps == 0:
        parser.error("--compile 至少需要 1 轮完整的前向/反向预热")
    if args.baseline is not None:
        if not args.compile:
            parser.error("--baseline 与 --compile 一起使用")
        if args.output is not None and args.output.resolve() == args.baseline.resolve():
            parser.error("output 不能覆盖 baseline 文件")
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

    attention_fn = scaled_dot_product_attention
    if args.compile:
        result["phase"] = "compile_setup"
        # 20 组静态形状分别测试，避免累积重编译次数触发上限。
        # reset 和编译包装均在计时外；实际前向/反向编译由下面的预热触发。
        #先重置编译状态
        torch.compiler.reset()
        #参1:要优化的原始函数/模型，参2:使用 PyTorch 的 Inductor 后端优化计算图并生成执行代码，参3:要求整个函数可以被捕获成一张计算图。如果遇到导致计算图中断的操作，就报错，参数4:针对当前输入的固定形状生成专用代码。输入形状变化时，通常需要重新编译。
        attention_fn = torch.compile(scaled_dot_product_attention, backend="inductor", fullgraph=True, dynamic=False)

    with torch.enable_grad():
        for _ in range(args.warmup_steps):
            result["phase"] = "warmup_forward"
            output = attention_fn(q, k, v, mask)
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
            output = attention_fn(q, k, v, mask)
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


def validate_baseline(report: dict, baseline: dict) -> None:
    """兼容已有 eager JSON；不同 GPU 或测试配置的结果不进行加速比比较。"""
    if not report["config"].get("compile", False) or baseline["config"].get("compile", False):
        raise ValueError("比较要求当前结果启用 compile，baseline 使用未编译实现")
    if baseline.get("run_status", "complete") != "complete":
        raise ValueError("baseline 运行异常退出，请使用完整的基准结果；配置中的 OOM 记录可以保留")
    for key in ("batch_size", "device_name", "torch_version", "cuda_runtime_version", "cuda_matmul_allow_tf32", "attention_function"):
        if report[key] != baseline[key]:
            raise ValueError(f"baseline 的 {key} 不一致：{baseline[key]!r} != {report[key]!r}")
    for key in ("dtype", "causal", "steps", "warmup_steps", "seed"):
        if report["config"][key] != baseline["config"][key]:
            raise ValueError(f"baseline 的 {key} 不一致")
    expected = set(product(report["config"]["d_models"], report["config"]["seq_lens"]))
    actual = {(row["d_model"], row["seq_len"]) for row in baseline["results"]}
    if expected != actual or len(baseline["results"]) != len(actual):
        raise ValueError("baseline 的形状配置不匹配或存在重复记录")
    for row in baseline["results"]:
        if row["status"] not in ("ok", "oom"):
            raise ValueError("baseline 包含无效状态")
        if row["status"] == "ok" and any(row[f"{phase}_completed_steps"] != baseline["config"]["steps"] for phase in ("forward", "backward")):
            raise ValueError("baseline 包含未完成全部测量轮次的配置")


def comparison_table(report: dict, baseline: dict) -> str:
    validate_baseline(report, baseline)
    original = {(row["d_model"], row["seq_len"]): row for row in baseline["results"]}
    lines = [
        f"GPU: {report['device_name']}; batch={report['batch_size']}; dtype={report['config']['dtype']}; "
        f"causal={report['config']['causal']}; warmup={report['config']['warmup_steps']}; steps={report['config']['steps']}.",
        "",
        "| d_model | seq_len | eager forward (ms) | compiled forward (ms) | forward speedup | eager backward (ms) | compiled backward (ms) | backward speedup |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for compiled in report["results"]:
        eager = original[(compiled["d_model"], compiled["seq_len"])]
        cells = [str(compiled["d_model"]), str(compiled["seq_len"])]
        for phase in ("forward", "backward"):
            for row in (eager, compiled):
                cells.append(f"{row[f'{phase}_mean_ms']:.3f}" if row["status"] == "ok" else f"OOM ({row['phase']})")
            speedup = eager[f"{phase}_mean_ms"] / compiled[f"{phase}_mean_ms"] if eager["status"] == compiled["status"] == "ok" else None
            cells.append(f"{speedup:.2f}x" if speedup is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "加速比 = eager 平均耗时 / compiled 平均耗时。编译开销不计入；任一版本 OOM 时不计算加速比。"]
    return "\n".join(lines) + "\n"


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
        "execution_mode": "compiled" if args.compile else "eager",
        "compile_options": {"backend": "inductor", "fullgraph": True, "dynamic": False, "mode": "default"} if args.compile else None,
        "compile_scope": "independent static shape per case; compiler cache reset before warmup; compilation excluded" if args.compile else None,
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
    baseline = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        validate_baseline(report, baseline)
    print(f"device={report['device_name']} mode={report['execution_mode']} dtype={args.dtype} batch={BATCH_SIZE} causal={args.causal} warmup={args.warmup_steps} steps={args.steps}", flush=True)
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
    if baseline is not None:
        comparison = comparison_table(report, baseline)
        print("\n" + comparison, flush=True)
        if args.output is not None:
            comparison_path = args.output.with_suffix(".comparison.md")
            comparison_path.write_text(comparison, encoding="utf-8")
            print(f"编译前后对比表已保存到：{comparison_path}", flush=True)


if __name__ == "__main__":
    main()

"""作业 2.1.3(a)：使用我们自己的 Transformer 做端到端基准测试。

运行方式：uv run python -m cs336_systems.benchmark --mode train
仅前向模式也保留自动求导记录，测量的是训练前向，而不是推理。
"""

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from timeit import default_timer

import torch

from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.transformer_lm import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train"), default="train")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    # 默认采用题目中的 small 模型，各项超参数也可以单独指定。
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda", help="cuda、cuda:0 或 cpu；CPU 用于本地检查")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="可选：把配置和逐步耗时保存为 JSON")
    args = parser.parse_args()

    return args


def run_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """三种模式计时的范围不同，初始化和随机数据生成由外部负责。"""
    if mode not in ("forward", "forward_backward", "train"):
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "train" and optimizer is None:
        raise ValueError("train 模式需要优化器")

    if mode != "forward":
        # 放在计时范围内；每一步独立计算梯度，不跨步骤累积。
        model.zero_grad(set_to_none=True)

    #前向传播
    logits = model(inputs)

    if mode == "forward":
        return

    #计算损失，反向传播
    loss = cross_entropy(inputs=logits, targets=targets)
    loss.backward()
    if mode == "train" and optimizer is not None:
        optimizer.step()

#等待GPU完成工作
def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_steps(step: Callable[[], None], device: torch.device, warmup_steps: int, steps: int) -> list[float]:
    """返回每个测量步骤的秒数，GPU 完成执行后才停止计时。"""
    # 排除模型初始化和数据生成留下的异步 CUDA 工作；w=0 时也需要。
    synchronize(device)
    for _ in range(warmup_steps):
        step()
        synchronize(device)

    timings = []
    for _ in range(steps):
        start = default_timer()
        step()
        synchronize(device)
        timings.append(default_timer() - start)
    return timings


def benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr) if args.mode == "train" else None

    # 同一个随机批次用于预热和测量；采样、设备传输不计入耗时。
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length + 1), device=device)
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    def step() -> None:
        run_step(model, inputs, targets, args.mode, optimizer)

    # 保证 forward 模式也记录训练所需的计算图。
    with torch.enable_grad():
        timings = measure_steps(step, device, args.warmup_steps, args.steps)

    config = {key: value for key, value in vars(args).items() if key != "output"}
    return {
        "config": config,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "step_times_seconds": timings,
        "mean_seconds": statistics.mean(timings),
        "std_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "std_convention": "sample (n-1); 0 for a single measurement",
    }


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    print(f"model={result['model_class']} parameters={result['parameter_count']:,}")
    print(f"device={result['device_name']} dtype={args.dtype} mode={args.mode}")
    print(f"warmup_steps={args.warmup_steps} measured_steps={args.steps}")
    for index, seconds in enumerate(result["step_times_seconds"], start=1):
        print(f"step={index} elapsed_ms={seconds * 1000:.3f}")
    print(f"mean_ms={result['mean_seconds'] * 1000:.3f} std_ms={result['std_seconds'] * 1000:.3f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

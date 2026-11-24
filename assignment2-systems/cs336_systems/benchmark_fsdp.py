"""在双卡上训练 XL Transformer，并标记前向权重收集的 GPU 时间线。"""

import argparse
import json
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.transformer_lm import TransformerLM
from cs336_systems.fsdp import FSDP


class ProfiledFSDP(FSDP):
    """只在采样步添加 NVTX 标记，保持原有的 FSDP 预取和等待逻辑。"""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None):
        self.layer_names = {layer: name for name, layer in module.named_modules()}
        self.trace_forward = False
        super().__init__(module, compute_dtype)

    def _wrap_layer(self, layer, params):
        forward = layer.forward
        label = f"FSDP compute {self.layer_names[layer]}"

        @wraps(forward)
        def traced_forward(*args, **kwargs):
            with torch.cuda.nvtx.range(label) if self.trace_forward else nullcontext():
                return forward(*args, **kwargs)

        layer.forward = traced_forward
        super()._wrap_layer(layer, params)

    def _gather_layer(self, layer):
        if self.trace_forward:
            with torch.cuda.nvtx.range(f"FSDP gather {self.layer_names[layer]}"):
                return super()._gather_layer(layer)
        return super()._gather_layer(layer)

    def _prepare_layer(self, layer):
        if self.trace_forward:
            # 在计算流排入一个极短的地标 kernel，标记该流抵达当前层的时间。
            # 这里尚未等待权重。计算 kernel 本身的开始时间已包含等待。
            with torch.cuda.nvtx.range(f"FSDP demand {self.layer_names[layer]}"):
                torch.cuda._sleep(1)
        return super()._prepare_layer(layer)


def train_step(model, optimizer, inputs, targets, *, profile=False):
    optimizer.zero_grad(set_to_none=True)
    model.trace_forward = profile
    try:
        with torch.cuda.nvtx.range("FSDP forward") if profile else nullcontext():
            logits = model(inputs)
    finally:
        model.trace_forward = False
    loss = cross_entropy(logits, targets)
    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()
    return loss.detach()


def benchmark_worker(rank, args):
    use_cuda = args.device == "cuda"
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    torch.set_num_threads(args.cpu_threads)
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    world_size = 2
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend, init_method=args.init_method, rank=rank, world_size=world_size, device_id=device if use_cuda else None)
    try:
        torch.manual_seed(args.seed)
        base = TransformerLM(
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
        parameter_count = sum(p.numel() for p in base.parameters())
        compute_dtype = torch.bfloat16 if args.dtype == "bf16" else None
        model = ProfiledFSDP(base, compute_dtype)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length + 1), device=device)
        local = tokens.chunk(world_size)[rank]
        inputs = local[:, :-1].contiguous()
        targets = local[:, 1:].contiguous()

        for _ in range(args.warmup_steps):
            train_step(model, optimizer, inputs, targets)
        dist.barrier()
        if use_cuda:
            torch.cuda.synchronize(device)
        times = []
        losses = []
        with torch.cuda.nvtx.range("measurement") if args.profile and rank == 0 else nullcontext():
            # 由 rank 0 打开采集区间后，两张卡才开始采样步骤。
            dist.barrier()
            if use_cuda:
                torch.cuda.synchronize(device)
            for _ in range(args.steps):
                start = perf_counter()
                loss = train_step(model, optimizer, inputs, targets, profile=args.profile)
                if use_cuda:
                    torch.cuda.synchronize(device)
                times.append(perf_counter() - start)
                losses.append(loss.item())
            timings = torch.tensor(times, device=device, dtype=torch.float64)
            dist.reduce(timings, dst=0, op=dist.ReduceOp.MAX)
            if rank == 0:
                # 在结束 NVTX 采集前写完。个别 Nsight 版本关闭采集时会终止被测子进程。
                result = {
                    "config": {k: v for k, v in vars(args).items() if k not in ("output", "init_method")},
                    "device_name": torch.cuda.get_device_name(device) if use_cuda else "CPU",
                    "backend": backend,
                    "world_size": world_size,
                    "local_batch_size": inputs.shape[0],
                    "parameter_count": parameter_count,
                    "forward_layer_order": [model.layer_names[layer] for layer in model._forward_order],
                    "step_times_seconds": timings.tolist(),
                    "rank_0_losses": losses,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"{result['device_name']}: {len(result['forward_layer_order'])} sharded layers, step times {timings.tolist()}", flush=True)
            dist.barrier()
            if use_cuda:
                torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--init-method", default="tcp://127.0.0.1:12391")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("--batch-size 必须能均分给两个 rank")
    if args.steps < 1 or args.warmup_steps < 0:
        parser.error("测量至少 1 步，预热步数不能为负数")
    if args.profile and (args.device != "cuda" or args.steps != 1 or args.warmup_steps < 1):
        parser.error("Nsight 采集需要 CUDA、至少一轮预热和单步测量")
    mp.spawn(benchmark_worker, args=(args,), nprocs=2, join=True)


if __name__ == "__main__":
    main()

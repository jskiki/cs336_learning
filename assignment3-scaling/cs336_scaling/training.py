"""PyTorch training and evaluation for the single-GPU DCLM pilot."""

import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .torch_model import FusedAdamW, TransformerConfig, TransformerLM

BATCH_SIZE = 8
CONTEXT_LENGTH = 512
TRAIN_STEPS = 200
WARMUP_STEPS = 10
EVAL_EVERY = 50
VAL_STEPS = 8
SEED = 42
LEARNING_RATE = 3e-4


def load_batches(
    path: Path, available_tokens: int, steps: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    used_tokens = steps * BATCH_SIZE * CONTEXT_LENGTH
    if path.stat().st_size != available_tokens * 2:
        raise ValueError(f"{path.name} size does not match the manifest")
    if used_tokens + 1 > available_tokens:
        raise ValueError(
            f"{path.name} needs {used_tokens + 1:,} tokens, but has {available_tokens:,}"
        )

    mapped = np.memmap(path, dtype="<u2", mode="r", shape=(used_tokens + 1,))
    tokens = torch.from_numpy(np.array(mapped, dtype=np.int64)).to(device)
    shape = (steps, BATCH_SIZE, CONTEXT_LENGTH)
    return tokens[:-1].view(shape), tokens[1:].view(shape)


def run_dclm_pilot(data_dir: Path) -> dict:
    """Train and evaluate on the prepared DCLM sample, scoped to one CUDA GPU."""
    import triton

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pilot needs a CUDA GPU with BF16 support")
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["dtype"] != "uint16-le":
        raise ValueError(f"Unexpected DCLM token format: {manifest['dtype']}")
    train_inputs, train_targets = load_batches(
        data_dir / "train.bin", manifest["tokens"]["train"], TRAIN_STEPS, device
    )
    val_inputs, val_targets = load_batches(
        data_dir / "val.bin", manifest["tokens"]["val"], VAL_STEPS, device
    )

    config = TransformerConfig(
        vocab_size=manifest["vocab_size"],
        context_length=CONTEXT_LENGTH,
        d_model=256,
        d_ff=768,
        num_layers=4,
        num_heads=4,
    )
    model = TransformerLM(config, device=device, dtype=torch.bfloat16)
    optimizer = FusedAdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        state_dtype=torch.float32,
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{torch.cuda.get_device_name()}: {params:,} parameters, {TRAIN_STEPS} training steps",
        flush=True,
    )

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses = [
            model(inputs, targets=targets)
            for inputs, targets in zip(val_inputs, val_targets, strict=True)
        ]
        mean_loss = torch.stack(losses).mean().item()
        model.train()
        if not math.isfinite(mean_loss):
            raise RuntimeError("Non-finite validation loss")
        return mean_loss

    history = [{"step": 0, "val_loss": evaluate()}]
    print(json.dumps(history[-1]), flush=True)
    measured_seconds = 0.0
    previous = 0
    for end in [WARMUP_STEPS, *range(EVAL_EVERY, TRAIN_STEPS + 1, EVAL_EVERY)]:
        torch.cuda.synchronize()
        start = perf_counter()
        losses = []
        for step in range(previous, end):
            optimizer.zero_grad(set_to_none=True)
            loss = model(train_inputs[step], targets=train_targets[step])
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        train_loss = torch.stack(losses).mean().item()
        if not math.isfinite(train_loss):
            raise RuntimeError(f"Non-finite training loss by step {end}")

        record = {"step": end, "train_loss": train_loss, "train_seconds": seconds}
        if end > WARMUP_STEPS:
            measured_seconds += seconds
            record["val_loss"] = evaluate()
        history.append(record)
        print(json.dumps(record), flush=True)
        previous = end

    tokens_per_step = BATCH_SIZE * CONTEXT_LENGTH
    return {
        "dataset": manifest["source"],
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "triton_version": triton.__version__,
        "model": asdict(config),
        "parameters": params,
        "seed": SEED,
        "optimizer": {
            "name": "FusedAdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.01,
            "state_dtype": "float32",
        },
        "batch_size": BATCH_SIZE,
        "train_tokens": TRAIN_STEPS * tokens_per_step,
        "val_tokens": VAL_STEPS * tokens_per_step,
        "steady_train_tokens_per_second": (TRAIN_STEPS - WARMUP_STEPS)
        * tokens_per_step
        / measured_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "history": history,
    }

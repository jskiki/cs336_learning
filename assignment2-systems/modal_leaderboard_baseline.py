"""在 Modal 2×B200 上运行原模型结构 + 未优化原版 FlashAttention2Triton 的双方案（ddp_sharded / fsdp）对照基线。

示例：
    uv run modal run modal_leaderboard_baseline.py --output results/leaderboard-baseline-compare.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import modal

DEFAULT_GPU = "B200:2"
PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/opt/cs336-systems"
SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store", "**/.venv/**"]

app = modal.App("cs336-leaderboard-baseline")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.13.6")
    .add_local_file(PROJECT_DIR / "pyproject.toml", f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .add_local_file(PROJECT_DIR / "uv.lock", f"{REMOTE_PROJECT_DIR}/uv.lock", copy=True)
    .add_local_dir(PROJECT_DIR / "cs336-basics", f"{REMOTE_PROJECT_DIR}/cs336-basics", copy=True, ignore=SOURCE_IGNORE)
    .run_commands(f"uv sync --project {REMOTE_PROJECT_DIR} --frozen --no-install-project")
    .add_local_dir(PROJECT_DIR / "cs336_systems", "/root/cs336_systems", ignore=SOURCE_IGNORE)
)

REMOTE_PYTHON = f"{REMOTE_PROJECT_DIR}/.venv/bin/python"


@app.function(image=image, gpu=DEFAULT_GPU, cpu=16, memory=64 * 1024, timeout=15 * 60)
def run_baseline_compare(extra_args: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="cs336-lb-base-") as directory:
        for idx, parallel in enumerate(("ddp_sharded", "fsdp")):
            output = Path(directory) / f"{parallel}.json"
            port = 29581 + idx
            cmd = [
                REMOTE_PYTHON,
                "-u",
                "-m",
                "cs336_systems.benchmark_leaderboard_baseline",
                *extra_args,
                "--parallel",
                parallel,
                "--init-method",
                f"tcp://127.0.0.1:{port}",
                "--output",
                str(output),
            ]
            proc = subprocess.run(cmd, cwd="/root", check=False)
            if output.is_file():
                results[parallel] = json.loads(output.read_text(encoding="utf-8"))
            else:
                results[parallel] = {"status": "failed", "parallel": parallel, "returncode": proc.returncode}
    return results


@app.local_entrypoint()
def main(*arguments):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--output", type=Path, default=Path("results/leaderboard-baseline-compare.json"))
    args, passthrough = parser.parse_known_args(arguments)

    target = run_baseline_compare if args.gpu == DEFAULT_GPU else run_baseline_compare.with_options(gpu=args.gpu)
    results = target.remote(passthrough)

    print("\n| 基线并行策略（原模型 + 未优化 FA2） | 状态 | median (ms) | mean (ms) | 峰值 Allocated (GiB) | 峰值 Reserved (GiB) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for mode, sample in results.items():
        if sample.get("status") == "ok":
            alloc = max(sample["peak_allocated_gib_by_rank"])
            resv = max(sample["peak_reserved_gib_by_rank"])
            print(f"| {mode} | ok | {sample['median_ms']:.2f} | {sample['mean_ms']:.2f} | {alloc:.2f} | {resv:.2f} |")
        else:
            print(f"| {mode} | {sample.get('status')} | — | — | {sample.get('rank0_peak_allocated_gib', '—')} | {sample.get('rank0_peak_reserved_gib', '—')} |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果已保存到 {args.output}")

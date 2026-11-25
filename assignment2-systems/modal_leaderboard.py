"""在 Modal 2×B200 上一键运行 Leaderboard 完整单测与 8B/32K 训练步基准。

常用命令（在 assignment2-systems 目录执行）：
    # 1. 在真 GPU 上跑全部 Triton / FusedAdamW / Leaderboard 正确性单测
    uv run modal run modal_leaderboard.py --mode test

    # 2. 快速验证 8B / 32K / 2×B200 端到端跑通并计时（warmup=2s, rep=5s）
    uv run modal run modal_leaderboard.py --mode bench --quick --verify --output results/leaderboard-quick.json

    # 3. 完整对齐题目计时（warmup=10s, rep=30s）
    uv run modal run modal_leaderboard.py --mode bench --parallel ddp_sharded --checkpoint-mode ffn --output results/leaderboard-b200.json

    # 4. 同容器对比 ddp_sharded vs fsdp
    uv run modal run modal_leaderboard.py --mode compare --quick --output results/leaderboard-compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal

DEFAULT_GPU = "B200:2"
PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/opt/cs336-systems"
SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store", "**/.venv/**"]

app = modal.App("cs336-leaderboard-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.13.6")
    .add_local_file(PROJECT_DIR / "pyproject.toml", f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .add_local_file(PROJECT_DIR / "uv.lock", f"{REMOTE_PROJECT_DIR}/uv.lock", copy=True)
    .add_local_dir(PROJECT_DIR / "cs336-basics", f"{REMOTE_PROJECT_DIR}/cs336-basics", copy=True, ignore=SOURCE_IGNORE)
    .run_commands(f"uv sync --project {REMOTE_PROJECT_DIR} --frozen --no-install-project")
    .add_local_dir(PROJECT_DIR / "cs336_systems", "/root/cs336_systems", ignore=SOURCE_IGNORE)
    .add_local_dir(PROJECT_DIR / "tests", "/root/tests", ignore=SOURCE_IGNORE)
)

REMOTE_PYTHON = f"{REMOTE_PROJECT_DIR}/.venv/bin/python"


@app.function(image=image, gpu="B200:1", cpu=8, memory=32 * 1024, timeout=10 * 60)
def run_gpu_tests(extra_env: dict[str, str], pytest_args: list[str]) -> None:
    env = os.environ | extra_env
    subprocess.run([REMOTE_PYTHON, "-m", "pytest", "tests/test_attention.py", *pytest_args], cwd="/root", env=env, check=True)
    subprocess.run(
        [
            REMOTE_PYTHON,
            "-c",
            "import torch; from cs336_systems.benchmark_leaderboard import verify_equivalence; "
            "print('FP32:', verify_equivalence(torch.device('cuda'), torch.float32, compile_ops=True)); "
            "print('BF16:', verify_equivalence(torch.device('cuda'), torch.bfloat16, compile_ops=True))",
        ],
        cwd="/root",
        env=env,
        check=True,
    )


@app.function(image=image, gpu=DEFAULT_GPU, cpu=16, memory=64 * 1024, timeout=12 * 60)
def run_leaderboard_bench(arguments: list[str], extra_env: dict[str, str]) -> dict:
    env = os.environ | extra_env
    with tempfile.TemporaryDirectory(prefix="cs336-lb-") as directory:
        output = Path(directory) / "leaderboard.json"
        cmd = [
            REMOTE_PYTHON,
            "-u",
            "-m",
            "cs336_systems.benchmark_leaderboard",
            *arguments,
            "--output",
            str(output),
        ]
        subprocess.run(cmd, cwd="/root", env=env, check=True)
        if not output.is_file():
            return {"status": "verify_only_ok"}
        return json.loads(output.read_text(encoding="utf-8"))


@app.function(image=image, gpu=DEFAULT_GPU, cpu=16, memory=64 * 1024, timeout=12 * 60)
def run_leaderboard_compare(arguments: list[str], extra_env: dict[str, str]) -> dict[str, dict]:
    env = os.environ | extra_env
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="cs336-lb-cmp-") as directory:
        for idx, parallel in enumerate(("ddp_sharded", "fsdp")):
            output = Path(directory) / f"{parallel}.json"
            port = 29521 + idx
            cmd = [
                REMOTE_PYTHON,
                "-u",
                "-m",
                "cs336_systems.benchmark_leaderboard",
                *arguments,
                "--parallel",
                parallel,
                "--init-method",
                f"tcp://127.0.0.1:{port}",
                "--output",
                str(output),
            ]
            subprocess.run(cmd, cwd="/root", env=env, check=True)
            results[parallel] = json.loads(output.read_text(encoding="utf-8"))
    return results


@app.local_entrypoint()
def main(*arguments):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("test", "bench", "compare"), default="bench")
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="bench/compare 使用的 GPU，默认 B200:2")
    parser.add_argument("--autotune", action="store_true", help="设置 CS336_FLASH_AUTOTUNE=1")
    parser.add_argument("--no-tma", action="store_true", help="设置 CS336_FLASH_USE_TMA=0")
    parser.add_argument("--output", type=Path, default=None, help="可选本地保存 JSON 路径")
    args, passthrough = parser.parse_known_args(arguments)

    extra_env: dict[str, str] = {}
    if args.autotune:
        extra_env["CS336_FLASH_AUTOTUNE"] = "1"
    if args.no_tma:
        extra_env["CS336_FLASH_USE_TMA"] = "0"

    if args.mode == "test":
        run_gpu_tests.remote(extra_env, passthrough)
        print("GPU 单元测试全部通过！")
        return

    if args.mode == "compare":
        target = run_leaderboard_compare if args.gpu == DEFAULT_GPU else run_leaderboard_compare.with_options(gpu=args.gpu)
        results = target.remote(passthrough, extra_env)
        print("\n| 并行策略 | median (ms) | mean (ms) | 峰值显存 (GiB) | 冷启动总耗时 (s) |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for mode, sample in results.items():
            peak = max(sample["peak_allocated_gib_by_rank"])
            print(f"| {mode} | {sample['median_ms']:.2f} | {sample['mean_ms']:.2f} | {peak:.2f} | {sample['total_wall_seconds']:.1f} |")
        result_to_save: dict = results
    else:
        target = run_leaderboard_bench if args.gpu == DEFAULT_GPU else run_leaderboard_bench.with_options(gpu=args.gpu)
        result_to_save = target.remote(passthrough, extra_env)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result_to_save, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"结果已保存到 {args.output}")

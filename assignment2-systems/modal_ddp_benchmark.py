"""在 Modal 双卡容器上比较 DDP 同步方式，并可下载 JSON 结果。"""

import argparse
import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import modal


DEFAULT_GPU = "B200:2"
GRADIENT_SYNC_MODES = ("individual", "flat", "overlap")
PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/opt/cs336-systems"
SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store", "**/.venv/**"]

app = modal.App("cs336-naive-ddp-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.13.6")
    .add_local_file(PROJECT_DIR / "pyproject.toml", f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .add_local_file(PROJECT_DIR / "uv.lock", f"{REMOTE_PROJECT_DIR}/uv.lock", copy=True)
    .add_local_dir(PROJECT_DIR / "cs336-basics", f"{REMOTE_PROJECT_DIR}/cs336-basics", copy=True, ignore=SOURCE_IGNORE)
    .run_commands(f"uv sync --project {REMOTE_PROJECT_DIR} --frozen --no-install-project")
    .add_local_dir(PROJECT_DIR / "cs336_systems", "/root/cs336_systems", ignore=SOURCE_IGNORE)
)


@app.function(image=image, gpu=DEFAULT_GPU, cpu=8, memory=16 * 1024, timeout=300)
def run_ddp_benchmark(arguments: list[str], modes: Sequence[str]) -> dict:
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for mode in modes:
            output = Path(directory) / f"{mode}.json"
            subprocess.run(
                [
                    f"{REMOTE_PROJECT_DIR}/.venv/bin/python", "-u", "-m", "cs336_systems.benchmark_ddp",
                    *arguments, "--device", "cuda", "--gradient-sync", mode, "--output", str(output),
                ],
                cwd="/root",
                check=True,
            )
            results[mode] = json.loads(output.read_text(encoding="utf-8"))
    return results


@app.local_entrypoint()
def main(*arguments):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="Modal GPU 类型及数量")
    parser.add_argument("--gradient-sync", choices=(*GRADIENT_SYNC_MODES, "all"), default="individual", help="用 all 在同一容器依次测试三种模式")
    parser.add_argument("--output", type=Path, help="可选本地 JSON 结果路径")
    args, benchmark_arguments = parser.parse_known_args(arguments)
    compare = args.gradient_sync == "all"
    modes = list(GRADIENT_SYNC_MODES) if compare else [args.gradient_sync]
    results = run_ddp_benchmark.with_options(gpu=args.gpu).remote(benchmark_arguments, modes)
    if compare and all("mean_seconds" in sample for sample in results.values()):
        print("| 同步模式 | 单步总耗时 (ms，均值 ± 标准差) | 相对逐参数的加速比 |")
        print("| --- | ---: | ---: |")
        baseline = results["individual"]["mean_seconds"]
        for mode, sample in results.items():
            mean, std = sample["mean_seconds"], sample["std_seconds"]
            print(f"| {mode} | {mean * 1000:.3f} ± {std * 1000:.3f} | {baseline / mean:.3f}× |")
    if args.output is not None:
        result = results if compare else results[args.gradient_sync]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"结果已保存到 {args.output}")

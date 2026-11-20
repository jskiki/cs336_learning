"""在 Modal 上运行单机 GPU/NCCL all-reduce 基准测试。"""

import argparse
import subprocess
import sys
from pathlib import Path

import modal


DEFAULT_GPU = "A100-80GB:6"
PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_SCRIPT = "/root/benchmark_all_reduce.py"

app = modal.App("cs336-all-reduce-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.11.0", "numpy==2.4.4")
    .add_local_file(PROJECT_DIR / "cs336_systems" / "benchmark_all_reduce.py", REMOTE_SCRIPT)
)


@app.function(image=image, gpu=DEFAULT_GPU, cpu=6, timeout=300)
def run_all_reduce(arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-u", REMOTE_SCRIPT, *arguments, "--backend", "nccl"],
        check=True,
    )


@app.local_entrypoint()
def main(*arguments):
    #description=__doc__ 把文件顶部的文档字符串用作帮助说
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="Modal GPU 类型及数量")
    args, benchmark_arguments = parser.parse_known_args(arguments)
    run_all_reduce.with_options(gpu=args.gpu).remote(benchmark_arguments)

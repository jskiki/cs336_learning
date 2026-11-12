"""在 Modal B200 上运行基准测试，将结果保存到本地。

在 assignment2-systems 目录运行：
    uv run modal run modal_benchmark.py --mode train
    uv run modal run modal_benchmark.py --mode forward --output results/b200-forward.json

--gpu 和 --output 由这个入口处理，其余参数交给 cs336_systems.benchmark。
云端固定使用 CUDA；默认模型配置、预热步数和测量步数沿用 benchmark.py。
"""

import argparse
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal


PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/opt/cs336-systems"
REMOTE_PYTHON = f"{REMOTE_PROJECT_DIR}/.venv/bin/python"

app = modal.App("cs336-systems-benchmark")

# 显式复制项目配置和本地依赖，确保云端也能解析 ./cs336-basics。
# 不上传 reference、数据集、本地虚拟环境或已有测试结果。
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.13.6")
    .add_local_file(PROJECT_DIR / "pyproject.toml", f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .add_local_file(PROJECT_DIR / "uv.lock", f"{REMOTE_PROJECT_DIR}/uv.lock", copy=True)
    .add_local_dir(
        PROJECT_DIR / "cs336-basics",
        f"{REMOTE_PROJECT_DIR}/cs336-basics",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/.DS_Store", "**/.venv/**"],
    )
    # 安装锁文件中的依赖（包括我们自己的 basics 包）；根包随后以源码提供。
    .run_commands(f"uv sync --project {REMOTE_PROJECT_DIR} --frozen --no-install-project")
    .add_local_dir(
        PROJECT_DIR / "cs336_systems",
        "/root/cs336_systems",
        ignore=["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"],
    )
)


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_benchmark(arguments: list[str]) -> dict:
    """在 GPU 容器中调用现有脚本；返回结果，不依赖容器磁盘持久化。"""
    with tempfile.TemporaryDirectory(prefix="cs336-benchmark-") as directory:
        output_path = Path(directory) / "result.json"
        command = [
            REMOTE_PYTHON,
            "-u",
            "-m",
            "cs336_systems.benchmark",
            *arguments,
            # 放在最后，保证设备和临时输出位置由远程入口控制。
            "--device",
            "cuda",
            "--output",
            str(output_path),
        ]
        print("Running:", " ".join(command), flush=True)
        # 子进程使用项目虚拟环境，并从 /root 找到上传的 cs336_systems。
        # 不捕获 stdout/stderr，让逐步计时和错误信息显示在 Modal 日志中。
        subprocess.run(command, cwd="/root", check=True)
        return json.loads(output_path.read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(*arguments):
    parser = argparse.ArgumentParser(
        description="在 Modal GPU 上执行 benchmark.py，并将结果保存到本地。",
        epilog="其余参数直接转交给基准测试，例如 --mode train --warmup-steps 5 --steps 10。设备固定为 CUDA。",
        allow_abbrev=False,
    )
    parser.add_argument("--gpu", default="B200", help="Modal GPU 类型，默认为 B200")
    parser.add_argument("--output", type=Path, help="本地 JSON 路径；相对路径以当前工作目录为起点")
    args, benchmark_arguments = parser.parse_known_args(arguments)

    # 默认保存到项目内的 results/，加入时间戳和短 UUID，避免覆盖之前的实验。
    output_path = args.output
    if output_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output_path = PROJECT_DIR / "results" / f"modal-benchmark-{stamp}-{uuid4().hex[:8]}.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在提交基准测试，GPU：{args.gpu}", flush=True)
    print(f"完成后保存到本地：{output_path}", flush=True)
    result = run_benchmark.with_options(gpu=args.gpu).remote(benchmark_arguments)

    # 这里运行在本地；远程返回后，直接把同一份结果写入本地文件。
    result["modal_gpu"] = args.gpu
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"mode={result['config']['mode']} device={result['device_name']} "
        f"mean_ms={result['mean_seconds'] * 1000:.3f} "
        f"std_ms={result['std_seconds'] * 1000:.3f}"
    )
    print(f"结果已保存到本地：{output_path}")

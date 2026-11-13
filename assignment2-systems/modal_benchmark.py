"""在 Modal B200 上运行基准测试，将结果保存到本地。

在 assignment2-systems 目录运行：
    uv run modal run modal_benchmark.py --mode train
    uv run modal run modal_benchmark.py --mode forward --output results/b200-forward.json
    uv run modal run modal_benchmark.py --profile --mode train --output results/nsys-small-train.json

--gpu、--output 和 --profile 由这个入口处理，其余参数交给 cs336_systems.benchmark。
云端固定使用 CUDA；默认模型配置、预热步数和测量步数沿用 benchmark.py。
--profile 只采集 measurement 区间，将 JSON 和同名 .nsys-rep 一起保存到本地。
"""

import argparse
import json
import os
import subprocess  # 运行外部命令或程序
import tempfile  # 创建临时文件或临时目录
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal


PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/opt/cs336-systems"
REMOTE_PYTHON = f"{REMOTE_PROJECT_DIR}/.venv/bin/python"
NSYS_PACKAGE_URL = "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2026.6.1.218-39266636.deb"
#报告分块传回
REPORT_CHUNK_BYTES = 1024 * 1024

app = modal.App("cs336-systems-benchmark")

# 显式复制项目配置和本地依赖，确保云端也能解析 ./cs336-basics。
# 不上传 reference、数据集、本地虚拟环境或已有测试结果。
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    # 使用 NVIDIA 公开仓库中的固定版本 CLI，避免安装完整桌面应用。
    .run_commands(
        f"curl -fL --retry 3 {NSYS_PACKAGE_URL} -o /tmp/nsight-systems-cli.deb",
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends /tmp/nsight-systems-cli.deb",
        "rm /tmp/nsight-systems-cli.deb",
        "nsys --version",
    )
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


def benchmark_command(arguments: list[str], output_path: Path) -> list[str]:
    return [
        REMOTE_PYTHON,
        "-u",
        "-m",
        "cs336_systems.benchmark",
        *arguments,
        "--device",
        "cuda",
        "--output",
        str(output_path),
    ]


def profile_command(command: list[str], report_prefix: Path) -> list[str]:
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,cublas,cudnn",
        # 容器内只采集 API/内核轨迹，关闭依赖 perf 权限的 CPU 采样。
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=nvtx",
        "--nvtx-capture=measurement",
        # 停止采集后继续运行 Python，使其完成 JSON 输出。
        "--capture-range-end=stop",
        f"--output={report_prefix}",
        "--",
        *command,
    ]


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_benchmark(arguments: list[str], profile: bool = False) -> Iterator[dict]:
    """先返回指标，再按 1 MiB 分块传回报告，不依赖容器磁盘持久化。"""
    #--timing-breakdown，是分段计时
    if profile and "--timing-breakdown" in arguments:
        raise ValueError("Nsight 采集请使用普通运行路径，不加 --timing-breakdown")
    with tempfile.TemporaryDirectory(prefix="cs336-benchmark-") as directory:
        output_path = Path(directory) / "result.json"
        report_prefix = Path(directory) / "profile"
        report_path = report_prefix.with_suffix(".nsys-rep")
        command = benchmark_command(arguments, output_path)
        env = os.environ.copy()
        nsys_version = None
        if profile:
            nsys_version = subprocess.check_output(["nsys", "--version"], text=True).strip()
            command = profile_command(command, report_prefix)
            # torch.cuda.nvtx 使用普通字符串，允许 nsys 匹配 measurement。
            env["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
        print("Running:", " ".join(command), flush=True)
        # 子进程使用项目虚拟环境，并从 /root 找到上传的 cs336_systems。
        # 不捕获 stdout/stderr，让逐步计时和错误信息显示在 Modal 日志中。
        subprocess.run(command, cwd="/root", env=env, check=True)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["profiled"] = profile
        report_size = 0
        if profile:
            if not report_path.is_file() or report_path.stat().st_size == 0:
                raise RuntimeError("nsys 没有生成报告，请检查云端采集日志和 measurement 标签")
            report_size = report_path.stat().st_size
            result["nsys"] = {"version": nsys_version, "capture_range": "measurement", "trace": "cuda,nvtx,cublas,cudnn"}
        yield {"kind": "result", "result": result, "report_size": report_size}
        if profile:
            with report_path.open("rb") as report_file:
                while chunk := report_file.read(REPORT_CHUNK_BYTES):
                    yield {"kind": "report_chunk", "data": chunk}


@app.local_entrypoint()
def main(*arguments):
    parser = argparse.ArgumentParser(
        description="在 Modal GPU 上执行 benchmark.py，并将结果保存到本地。",
        epilog="其余参数直接转交给基准测试，例如 --mode train --warmup-steps 5 --steps 10。设备固定为 CUDA。",
        allow_abbrev=False,
    )
    parser.add_argument("--gpu", default="B200", help="Modal GPU 类型，默认为 B200")
    parser.add_argument("--output", type=Path, help="本地 JSON 路径；相对路径以当前工作目录为起点")
    parser.add_argument("--profile", action="store_true", help="用 Nsight Systems 采集正式测量区间，并下载同名 .nsys-rep 报告")
    args, benchmark_arguments = parser.parse_known_args(arguments)
    if args.profile and "--timing-breakdown" in benchmark_arguments:
        parser.error("--profile 不与 --timing-breakdown 同用；普通路径已有 NVTX 标签")

    # 默认保存到项目内的 results/，加入时间戳和短 UUID，避免覆盖之前的实验。
    output_path = args.output
    if output_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        kind = "profile" if args.profile else "benchmark"
        output_path = PROJECT_DIR / "results" / f"modal-{kind}-{stamp}-{uuid4().hex[:8]}.json"
    output_path = output_path.expanduser().resolve()
    if args.profile and output_path.suffix.lower() != ".json":
        parser.error("--profile 的 --output 请使用 .json 后缀；同名 .nsys-rep 会自动下载")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在提交基准测试，GPU：{args.gpu}", flush=True)
    print(f"完成后保存到本地：{output_path}", flush=True)
    report_path = output_path.with_suffix(".nsys-rep")
    partial_path = report_path.with_name(report_path.name + ".part")
    if args.profile:
        print(f"Nsight 报告：{report_path}；采集耗时包含分析工具开销。", flush=True)
    target = run_benchmark if args.gpu == "B200" else run_benchmark.with_options(gpu=args.gpu)
    result = None
    expected_size = 0
    received_size = 0
    # 使用 .part 临时文件，传输完整后再改名，避免把中断下载当成有效报告。
    with partial_path.open("wb") if args.profile else nullcontext() as report_file:
        for packet in target.remote_gen(benchmark_arguments, args.profile):
            if packet["kind"] == "result":
                result = packet["result"]
                expected_size = packet["report_size"]
            elif packet["kind"] == "report_chunk" and report_file is not None:
                received_size += report_file.write(packet["data"])
            else:
                raise RuntimeError("收到未知的云端结果类型")
    if result is None:
        raise RuntimeError("云端没有返回基准测试指标")
    if args.profile:
        if expected_size <= 0 or received_size != expected_size:
            raise RuntimeError(f"报告传输不完整：收到 {received_size} / {expected_size} 字节，临时文件位于 {partial_path}")
        partial_path.replace(report_path)
        result["nsys"]["local_report"] = str(report_path)

    # 这里运行在本地；远程返回后，直接把同一份结果写入本地文件。
    result["modal_gpu"] = args.gpu
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"mode={result['config']['mode']} device={result['device_name']} mean_ms={result['mean_seconds'] * 1000:.3f} std_ms={result['std_seconds'] * 1000:.3f}")
    print(f"结果已保存到本地：{output_path}")
    if args.profile:
        print(f"Nsight 报告已保存：{report_path}")

"""提交双卡 FSDP 基准与 Nsight 采集，保存 JSON、原始报告和 SQLite。"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

from modal_benchmark import REMOTE_PYTHON, REPORT_CHUNK_BYTES, image, profile_command


app = modal.App("cs336-fsdp-profile")
ARTIFACT_NAMES = ("fsdp-xl.json", "fsdp-xl.nsys-rep", "fsdp-xl.sqlite")


@app.function(image=image.add_local_python_source("modal_benchmark"), gpu="B200:2", cpu=8, memory=16 * 1024, timeout=900)
def profile_fsdp(arguments: list[str]):
    with TemporaryDirectory(prefix="cs336-fsdp-") as directory:
        output = Path(directory)
        stats, report, database = (output / name for name in ARTIFACT_NAMES)
        command = [REMOTE_PYTHON, "-u", "-m", "cs336_systems.benchmark_fsdp", *arguments, "--profile", "--output", str(stats)]
        env = os.environ | {"NSYS_NVTX_PROFILER_REGISTER_ONLY": "0"}
        subprocess.run(profile_command(command, report.with_suffix("")), cwd="/root", env=env, check=True)
        subprocess.run(["nsys", "export", "--type=sqlite", f"--output={database}", str(report)], check=True)
        result = json.loads(stats.read_text(encoding="utf-8"))
        result["nsys_version"] = subprocess.check_output(["nsys", "--version"], text=True).strip()
        stats.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        for path in (stats, report, database):
            yield (path.name, path.stat().st_size)
            with path.open("rb") as artifact:
                while chunk := artifact.read(REPORT_CHUNK_BYTES):
                    yield (path.name, chunk)


@app.local_entrypoint()
def main(*arguments):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=Path("results/b200-fsdp/nsight"))
    args, benchmark_arguments = parser.parse_known_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=args.output_dir) as directory:
        temporary = Path(directory)
        sizes = {}
        for name, value in profile_fsdp.remote_gen(benchmark_arguments):
            path = temporary / name
            if isinstance(value, int):
                sizes[name] = value
            else:
                with path.open("ab") as artifact:
                    artifact.write(value)
        for name in ARTIFACT_NAMES:
            artifact = temporary / name
            if not artifact.is_file() or not sizes.get(name) or artifact.stat().st_size != sizes[name]:
                raise RuntimeError(f"下载不完整：{name}")
        for name in ARTIFACT_NAMES:
            output = args.output_dir / name
            (temporary / name).replace(output)
            print(f"已保存 {output}")

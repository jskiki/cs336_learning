"""在 Modal 上运行注意力基准测试，并将 JSON 和 Markdown 表格保存到本地。

在 assignment2-systems 目录运行：
    uv run modal run modal_attention_benchmark.py
    uv run modal run modal_attention_benchmark.py --gpu A100-40GB --output results/attention.json
    uv run modal run modal_attention_benchmark.py --gpu A100-80GB --compile --baseline results/pytorch-attention-a100-80gb.json --output results/compiled-attention-a100-80gb.json

默认 batch=8、FP32、无 mask，预热 5 轮、正式测量 100 轮，覆盖题目全部配置。
除了 --gpu、--output 和 --baseline，其余参数传递给 cs336_systems.benchmark_attention。
baseline 在本地读取，与下载的 compiled 结果生成 .comparison.md，不上传已有结果文件。
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
app = modal.App("cs336-attention-benchmark")
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
    .run_commands(f"uv sync --project {REMOTE_PROJECT_DIR} --frozen --no-install-project")
    .add_local_dir(
        PROJECT_DIR / "cs336_systems",
        "/root/cs336_systems",
        ignore=["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"],
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=15 * 60)
def run_attention(arguments: list[str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="attention-benchmark-") as directory:
        output = Path(directory) / "result.json"
        command = [
            f"{REMOTE_PROJECT_DIR}/.venv/bin/python", "-u", "-m",
            "cs336_systems.benchmark_attention", *arguments,
            "--device", "cuda", "--output", str(output),
        ]
        completed = subprocess.run(command, cwd="/root", check=False)
        if not output.is_file():
            raise RuntimeError(f"注意力基准测试未产生结果，退出码：{completed.returncode}；请查看 Modal 日志")
        result = json.loads(output.read_text(encoding="utf-8"))
        result["returncode"] = completed.returncode
        result["run_status"] = "complete" if completed.returncode == 0 else "failed"
        return result


@app.local_entrypoint()
def main(*arguments):
    from cs336_systems.benchmark_attention import comparison_table, table_row

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gpu", default="A100-40GB")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, help="本地 eager JSON 结果，用于生成编译前后对比表")
    args, benchmark_arguments = parser.parse_known_args(arguments)
    baseline = None
    if args.baseline is not None:
        if "--compile" not in benchmark_arguments:
            parser.error("--baseline 与 --compile 一起使用")
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline["config"].get("compile", False):
            parser.error("baseline 必须是未编译的测试结果")
    output = args.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output = PROJECT_DIR / "results" / f"modal-attention-{stamp}-{uuid4().hex[:8]}.json"
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".json":
        parser.error("--output 请使用 .json 后缀，同时生成同名 .md 表格")
    if args.baseline is not None and output == args.baseline.resolve():
        parser.error("output 不能覆盖 baseline 文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {args.gpu}; 本地结果: {output}", flush=True)
    target = run_attention if args.gpu == "A100-40GB" else run_attention.with_options(gpu=args.gpu)
    result = target.remote(benchmark_arguments)
    result["modal_gpu_requested"] = args.gpu
    if args.baseline is not None:
        result["comparison_baseline"] = str(args.baseline.resolve())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = result["config"]
    lines = [
        f"GPU: {result['device_name']}; compiled={config.get('compile', False)}; batch={result['batch_size']}; dtype={config['dtype']}; "
        f"causal={config['causal']}; warmup={config['warmup_steps']}; steps={config['steps']}.",
        "",
        "| d_model | seq_len | status | forward mean (ms) | backward mean (ms) | before backward (MiB) | forward live delta (MiB) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *(table_row(row) for row in result["results"]),
        "",
        "显存列为正式测量中反向传播开始前的最大 allocated 值；forward live delta 包含输出和保存的张量。",
        "OOM 行的阶段标注保留在 status 中；原始计时、显存、峰值和错误详情见同名 JSON。",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已保存 JSON：{output}\n已保存表格：{output.with_suffix('.md')}", flush=True)
    if result["run_status"] != "complete":
        raise RuntimeError("云端测试异常退出，已保存部分结果；请查看 Modal 日志")
    if baseline is not None:
        comparison_path = output.with_suffix(".comparison.md")
        comparison_path.write_text(comparison_table(result, baseline), encoding="utf-8")
        print(f"编译前后对比表已保存：{comparison_path}", flush=True)

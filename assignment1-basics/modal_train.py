"""使用 cs336-data 存储卷，在 Modal 云端运行现有训练程序。

在 assignment1-basics 目录运行：uv run modal run modal_train.py
启动参数示例：--gpu A10G --run-name experiment
其余参数转交给 cs336_basics.train，参数名沿用训练程序的下划线写法。
"""

from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
#用来生成通用唯一识别码 UUID（Universally Unique Identifier）
from uuid import uuid4

import modal

#创建一个modal应用对象，传入名称
app = modal.App("cs336-basics-train")
#获取名为 cs336-data 的已有存储卷的引用
volume = modal.Volume.from_name("cs336-data")
#描述云端运行环境，image是容器镜像
image = (
    #以精简的 Debian Linux 为基础，使用 Python 3.12
    modal.Image.debian_slim(python_version="3.12")
    #找到项目目录，根据其中的 pyproject.toml 和 uv.lock 安装云端依赖。
    .uv_sync(uv_project_dir=Path(__file__).parent.as_posix()) #.as_posix()转成使用 / 分隔的路径字符串
    #把本地 cs336_basics 包的 Python 源码提供给云端容器
    .add_local_python_source("cs336_basics")
)

#生成训练参数的函数，相当于在命令行输入了python的带参数的运行指令
def training_command(arguments: list[str], run_dir: str) -> list[str]:
    defaults = {
        "train_path": "/data/tinystories_train.bin",
        "val_path": "/data/tinystories_valid.bin",
        "vocab_size": 10000,
        "data_dtype": "uint16",
        "batch_size": 64,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "max_iters": 20,
        "warmup_iters": 2,
        "eval_batches": 2,
    }
    return [
        "-u",
        "-m",
        "cs336_basics.train",
        #把字典装换成参数的形式如从{"batch_size": 64, "max_iters": 20}->["--batch_size", "64", "--max_iters", "20"]
        *[argument for key, value in defaults.items() for argument in (f"--{key}", str(value))],
        *arguments,
        "--device",
        "cuda",
        "--checkpoint_path",
        f"{run_dir}/latest.pt",
    ]

#将后面的函数注册为云端函数，image：使用前面描述的容器环境，volumes={"/data": volume}：将cs336-data卷挂载到容器的/data目录，timeout：云端函数执行超时时间单位为秒
@app.function(image=image, gpu="B200", volumes={"/data": volume}, timeout=24 * 60 * 60)
def run_training(arguments: list[str], run_name: str) -> dict:#run_name本次训练的名称
    import json
    import math
    import subprocess
    import sys

    import torch

    #构造结果目录
    run_dir = Path("/data/runs") / run_name
    # 训练程序以写入模式打开 metrics.jsonl，恢复训练时也会清空该文件。
    # 因此每次都使用新目录，保留已有日志和用于恢复训练的检查点。exist_ok=False：目标目录已经存在就报错
    run_dir.mkdir(exist_ok=False, parents=True)
    print(f"显卡：{torch.cuda.get_device_name()}", flush=True)
    print(f"结果保存目录：{run_dir}", flush=True)

    #下面的代码启动真正的训练，sys.executable：云端当前Python解释器的路径，training_command(...)：生成前面解释过的训练参数，*把这些参数展开到命令列表里
    #subprocess.run(...)：启动一个子进程，并等待它退出，result：保存子进程的退出码等执行结果，check=False：子进程退出码非零时，先不要立即抛异常（详见Note.md)
    result = subprocess.run([sys.executable, *training_command(arguments, run_dir.as_posix())], check=False)
    # 即使训练程序报错退出，也将已经生成的日志和检查点持久化到存储卷。
    volume.commit()
    result.check_returncode()

    #读取指标文件，检查是否有异常的训练指标（整个模型训练完检查）
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    if not all(math.isfinite(record[key]) for record in records for key in ("train_loss", "val_loss") if key in record):
        raise ValueError(f"{run_dir}/metrics.jsonl 中出现非有限损失值（NaN 或无穷大）")

    #读取本次训练配置
    config = json.loads((run_dir / "config.json").read_text())
    #读取检查点，检查是否达到要求的训练步数
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
    if checkpoint["iteration"] != config["max_iters"]:
        raise ValueError("检查点中保存的训练步数与指定的总训练步数不一致")
    
    #run_dir：结果目录，转换成字符串,checkpoint_step：检查点保存的训练步数,metrics：日志中最后三条记录
    return {"run_dir": run_dir.as_posix(), "checkpoint_step": checkpoint["iteration"], "metrics": records[-3:]}

#指定本地入口
@app.local_entrypoint()
def main(*arguments):
    import json

    parser = ArgumentParser(
        description="在 Modal 云端运行 cs336_basics.train，默认使用原来的 20 步冒烟训练配置。",
        epilog="其余参数转交给训练程序，例如 --max_iters 1000 --warmup_iters 100。",
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("--gpu", default="B200", help="Modal 显卡类型，默认为 B200")
    parser.add_argument("--run-name", default="", help="cs336-data/runs 下的新目录名；目录不能已存在，省略时自动生成")

    #把参数拆成两部分args：这个启动脚本认识的参数，例如 --gpu`、--run-name
    args, trainer_arguments = parser.parse_known_args(arguments)
    if args.run_name and (args.run_name in (".", "..") or "/" in args.run_name or "\\" in args.run_name):
        parser.error("--run-name 必须是单个目录名，不能为 .、.. 或包含路径分隔符")
    run_name = args.run_name or f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    print(f"正在提交训练，显卡：{args.gpu}；结果保存至：cs336-data/runs/{run_name}", flush=True)
    call = run_training.with_options(gpu=args.gpu).spawn(trainer_arguments, run_name)
    print(f"云端调用编号：{call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2))

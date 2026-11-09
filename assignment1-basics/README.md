# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 在 Modal 云端训练

在 `assignment1-basics` 目录执行以下命令，首次使用时通过 `uv run modal setup` 登录。
已有的 `cs336-data` 存储卷根目录中需要包含 `tinystories_train.bin` 和
`tinystories_valid.bin`。默认沿用原冒烟训练的配置：词表大小为 10,000，数据为
`uint16` 格式的原始二进制 token 序列；编码这些文件时使用的分词器必须与该词表匹配。

```powershell
# 如需代理，在当前终端设置本地 HTTP 或混合代理端口。
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:PYTHONUTF8 = "1"

# 使用 A10G 显卡，按原配置运行 20 步冒烟训练。
uv run modal run modal_train.py

# 延长训练；训练参数沿用 cs336_basics/train.py 中的下划线命名。
uv run modal run --detach modal_train.py --run-name tinystories-1k --max_iters 1000 --warmup_iters 100 --eval_batches 10

# 从原检查点恢复，结果写入新目录；目标总步数为 1000。
uv run modal run --detach modal_train.py --run-name smoke-resumed --resume_path /data/runs/smoke/latest.pt --max_iters 1000 --warmup_iters 100
```

启动脚本会上传当前的 `cs336_basics` Python 源码，并根据 `pyproject.toml` 和 `uv.lock`
构建云端依赖环境。它复用现有训练程序，默认批次大小为 64、上下文长度为 256、模型宽度为
512、层数为 4、注意力头数为 16、前馈网络宽度为 1344，参数精度为 float32。
可直接传入训练参数覆盖默认值，也可通过 `--gpu A100` 选择其他显卡。

恢复训练时，模型参数应与检查点一致。`--max_iters` 表示目标总步数，必须大于检查点中的
已完成步数才会继续训练。未单独指定余弦调度终点时，终点随 `--max_iters` 设置，因此延长
训练会改变学习率轨迹。现有训练程序没有恢复随机数生成器状态，恢复后的结果无法做到逐位复现。

每次运行都会在新的 `cs336-data/runs/<name>` 目录中保存 `config.json`、`metrics.jsonl`
和 `latest.pt`。省略目录名时会自动生成；如果目录已存在，脚本会拒绝写入，以保留已有日志和
检查点。CUDA 设备和输出检查点路径由启动脚本统一设置。训练程序退出时，已生成的结果会提交到
存储卷，即使训练报错也会保存部分产物。训练成功后，脚本会检查损失值是否有限、检查点步数是否
与目标总步数一致，然后打印最终指标。

```powershell
uv run modal volume ls cs336-data /runs
uv run modal volume get cs336-data /runs/<name>/metrics.jsonl -
```


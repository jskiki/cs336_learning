# CS336 作业工作区

个人 CS336 作业与学习笔记，用于在不同电脑之间同步学习进度。

## 在两台电脑之间同步学习进度

仓库地址：<https://github.com/jskiki/cs336_learning>

另一台电脑首次使用时，在希望存放项目的目录执行：

```powershell
git clone https://github.com/jskiki/cs336_learning.git
cd cs336_learning
cd assignment1-basics
uv sync --python 3.12
```

先安装 Git 和 uv。其他作业应按各自 README 单独安装环境；虚拟环境不随仓库复制。

以后在工作区根目录操作。开始学习前先拉取：

```powershell
git pull --ff-only
```

结束学习、准备换电脑前，检查并上传自己的修改：

```powershell
git status
git add --all
git commit -m "Update learning progress"
git push
```

没有修改时不需要提交。另一台电脑开始学习前同样先拉取。尽量不要让两台电脑在未同步时修改同一份文件；如果拉取提示本地修改或分支分叉，先保留修改并解决冲突，不要强制覆盖。

本仓库保留源代码、测试及其必要样例、题目 PDF、学习笔记、依赖配置和锁文件。各作业的 `data/` 目录、训练输出、W&B 日志、虚拟环境和缓存不上传；需要按对应作业说明重新下载或单独传输。测试目录内的小型模型和固定样例会保留，以便另一台电脑运行测试。

`.env` 和其他本机密钥文件不上传，需要时在每台电脑上分别配置。聊天中的解释只有保存为笔记文件后才会随 Git 同步。

已按 [Stanford CS336 Spring 2026 官网](https://cs336.stanford.edu/) 当前链接的五个官方仓库搭建。各作业保留原始题目、代码骨架、测试、依赖配置和官方说明，核心实现留给你完成。

## 目录导航

| 作业目录 | 主题 | 题目 | 官方说明 |
| --- | --- | --- | --- |
| [assignment1-basics](./assignment1-basics/) | 分词器、Transformer、优化器与训练 | [PDF](./assignment1-basics/cs336_assignment1_basics.pdf) | [README](./assignment1-basics/README.md) |
| [assignment2-systems](./assignment2-systems/) | 性能分析、GPU kernel、分布式训练 | [PDF](./assignment2-systems/cs336_assignment2_systems.pdf) | [README](./assignment2-systems/README.md) |
| [assignment3-scaling](./assignment3-scaling/) | Scaling Law 与训练 API | [PDF](./assignment3-scaling/cs336_assignment3_scaling.pdf) | [README](./assignment3-scaling/README.md) |
| [assignment4-data](./assignment4-data/) | 预训练数据处理、过滤与去重 | [PDF](./assignment4-data/cs336_assignment4_data.pdf) | [README](./assignment4-data/README.md) |
| [assignment5-alignment](./assignment5-alignment/) | 监督微调与强化学习 | [PDF](./assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf)、[选做部分](./assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | [README](./assignment5-alignment/README.md) |

## 从 A1 开始

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。以下命令在本工作区根目录的 PowerShell 中执行：

```powershell
cd .\assignment1-basics
uv sync --python 3.12
uv run pytest
```

`uv sync` 会下载 Python（如需要）和项目依赖，并在该作业目录创建 `.venv`。初次安装包含 PyTorch，可能需要较长时间和较大下载量。

A1 的测试通过 `tests/adapters.py` 调用你的实现；开始时出现 `NotImplementedError` 是官方骨架的预期状态。阅读 PDF 后，在 `cs336_basics/` 中逐步实现，再连接到适配接口。

## 环境与版本

- 每份作业独立管理环境：先进入对应目录，再按该目录 README 使用 `uv`。不同作业的 Python 和依赖约束不同，不要把五份作业合并成一个环境。
- A1 可以先在 CPU 上进行实现与正确性检查；A2 的 GPU kernel 和 GPU 实验等部分需要相应的 GPU/Linux 环境。后续 GPU 部分可使用 WSL2、Linux 服务器或云端环境，具体以各题目要求为准。
- A5 的官方 `tool.uv.environments` 仅列出 Linux x86_64 与 macOS ARM64，且要求 Python 3.12；Windows 下建议在 WSL2/Linux 中使用。GPU 依赖定义在 `gpu` extra 中，安装方式需结合该目录的 `pyproject.toml` 和 README 查看。
- A3 官方 README 区分 Stanford 学生和自学者的使用方式；访问课程训练 API 需要对应凭据，本地部署方式见其 README。
- A1 官方 README 标题仍写着 Spring 2025，但本次下载的项目版本为 `26.0.0`；这里保留上游文件原样。具体来源、版本和下载包摘要记录在 [sources.json](./sources.json)。
- 本次使用 GitHub 官方 `main` 分支压缩包导入，目录中不含 Git 历史。后续更新时应先备份自己的实现，避免直接覆盖。

## 本次搭建的验证范围

已核对五份下载包与解压后文件的一致性，以及题目、项目配置和测试目录是否齐全。当前终端找不到 `python` 和 `uv`，因此尚未安装依赖、执行 pytest 或启动训练。数据集和模型权重需按各作业说明另行下载。

"""作业 5.1：数据加载框架，核心采样逻辑由你完成。

测试入口：tests/adapters.py 中的 run_get_batch。
完成后，在 assignment1-basics 目录运行：
    uv run pytest -k test_get_batch
"""
from numpy import arange
from numpy.typing import NDArray
from numpy.random import randint
from torch import Tensor,as_tensor,long


def get_batch(
    dataset: NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """从 token 序列中随机采样一批输入序列及其下一 token 目标。

    参数：
        dataset: 包含整数 token ID 的一维 NumPy 数组，也可以是 np.memmap。
        batch_size: 每批采样的序列数量。
        context_length: 每条输入序列及目标序列的长度。
        device: 返回张量所在的 PyTorch 设备，例如 'cpu'、'cuda:0' 或 'mps'。

    返回值：
        (inputs, targets)，两个张量的形状均为 (batch_size, context_length)，
        数据类型均为 torch.long，且都位于指定设备上。

    输入约定：
        batch_size > 0，context_length > 0。
        dataset 的长度至少为 context_length + 1。

    完成标准：
        每条输入序列都是数据集中的连续片段，目标是对应的下一 token。
        从所有合法起始位置均匀随机采样，允许重复采样同一位置。
        不修改原始数据，不将整个数据集转换为张量或搬到目标设备。
        数据集文件的内存映射加载由调用方负责，此函数只接收数组。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if context_length <= 0:
        raise ValueError("context_length 必须大于 0")
    if len(dataset) <= context_length:
        raise ValueError("dataset 的长度至少为 context_length + 1")
    # TODO 1：采样 batch_size 个合法起始位置。
    start_indices = randint(
        low=0,
        high=len(dataset) - context_length,
        size=batch_size,
    )
    # TODO 2：构造输入序列及其对应的下一 token 目标序列。
    indices = start_indices[:, None] + arange(context_length + 1)
    # TODO 3：将这一批数据转换为指定设备上的整数张量。
    #indice是二维的，里面的每一个数字都是索引，去一维的dataset里面取出对应的数据，as_tensor将数据转换为Pytorch张量还会尽可能的避免数据复制，如果数据类型一致，且目标设配为cpu就可以共享内存
    batch = as_tensor(dataset[indices], dtype=long, device=device)
    # TODO 4：返回 (inputs, targets)。
    #构造输入和目标
    return batch[:, :-1], batch[:, 1:]

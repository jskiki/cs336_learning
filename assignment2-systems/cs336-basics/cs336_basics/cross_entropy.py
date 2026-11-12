"""作业 4.1：数值稳定的交叉熵损失。

测试入口已接入 tests/adapters.py 中的 run_cross_entropy。
在 assignment1-basics 目录运行：uv run pytest -k test_cross_entropy。
"""

from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """根据预测 logits 和目标 token ID，返回平均交叉熵损失。

    参数：
        inputs: 未经 softmax 的浮点 logits，形状为 (..., vocab_size)。
        targets: 整数类别索引，形状为 (...)，与 inputs 的前导维度一致。
            每个索引在 [0, vocab_size) 内，与 inputs 位于同一设备。

    返回值：
        对所有前导维度取平均后的标量 Tensor，形状为 ()。
        例如 inputs 为 (batch_size, seq_len, vocab_size) 时，
        targets 为 (batch_size, seq_len)，每个 token 位置参与平均。

    完成标准：
        按题目要求减去最大元素，保证数值稳定性。
        尽可能消去相互抵消的 log 和 exp 运算。
        支持任意数量的前导批次维度，最后一维始终是词表维度。
        不原地修改输入，保留从损失到 inputs 的 autograd 计算图。
    """
    # 平移量只用于数值稳定；视为常量不会改变损失对 logits 的梯度。
    #detach():得到一个与inputs共享数据。但脱离自动求导计算图的张量。它不复制数据，也不会让原来的 inputs 停止计算梯度。keepdim保留维度进行广播
    shifted = inputs - inputs.detach().amax(dim=-1, keepdim=True)
    #targets.unsqueeze(-1):在最后一维添加一个维度。shifted.gather(dim=-1, index=...)：沿最后一维按索引取值。详解见note，squeeze(-1)去掉最后大小为1的维度
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    return (log_normalizer - target_logits).mean()

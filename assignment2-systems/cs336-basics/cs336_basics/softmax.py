"""作业 3.4.4：softmax 的接口框架，核心计算由你完成。"""

from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """沿指定维度应用数值稳定的 softmax。

    参数：
        in_features: 任意形状的浮点 tensor。
        dim: 要归一化的维度，支持负数索引。

    完成标准：
        输出形状、device 和 dtype 与输入一致。
        沿 dim 的元素非负，且总和约为 1。
        输入整体加上同一个常数后，输出应基本不变。
        不原地修改输入，并保留 autograd 计算图。
    """
    # TODO：按题目要求完成数值稳定处理和沿 dim 的归一化。
    # 注意 reduction 后的形状应能与输入进行 broadcasting。
    #假如in_features的形状(2,3,4),则in_features.amax(dim=dim,keepdim=True)->(2,1,4),keppdim,保留了中间那一维，之后相减又利用了广播机制
    weights = (
        in_features - in_features.amax(dim=dim,keepdim=True)
    ).exp_()
    #整个tensor都进行了计算，dim只是决定怎么分组
    return weights / weights.sum(dim=dim, keepdim=True)

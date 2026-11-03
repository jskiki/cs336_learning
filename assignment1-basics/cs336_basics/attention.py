"""作业 3.4.4：scaled dot-product attention 的接口框架。"""

from torch import Tensor
from math import sqrt
from einops import einsum

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """计算 attention，支持任意数量的前导 batch-like 维度。

    参数：
        Q: 浮点 tensor，形状为 (..., queries, d_k)。
        K: 浮点 tensor，形状为 (..., keys, d_k)。
        V: 浮点 tensor，形状为 (..., keys, d_v)。
        mask: 可选布尔 tensor，形状为 (queries, keys)，也支持带有
            可广播的前导 batch-like 维度。True 表示允许关注，False 表示屏蔽。

    返回：
        形状为 (..., queries, d_v) 的 tensor。

    完成标准：
        保留前导 batch-like 维度，不写死 batch 或 head 的数量。
        每个 query 在未屏蔽的 key 上的 attention probabilities 之和为 1，
        被屏蔽的位置概率为 0。
        不原地修改 Q、K、V 或 mask，并保留 autograd 计算图。
    """
    # TODO 1：计算缩放后的 attention scores，形状为 (..., queries, keys)。
    scores = einsum(Q,K,"... queries d_k,... keys d_k -> ... queries keys")/sqrt(Q.shape[-1])
    # TODO 2：如果提供了 mask，在 softmax 之前处理被屏蔽的位置。
    #fill后面的_表示原地修改scores，~mask表示将布尔值取反(~表示按位取反)
    if mask is not None:
        scores.masked_fill_(~mask ,float("-inf"))
    # TODO 3：调用已有的 softmax，沿 key 维度得到 attention probabilities。
    probabilities = softmax(scores,dim=-1)
    # TODO 4：结合 V 得到输出，形状为 (..., queries, d_v)。
    return einsum(probabilities,V,"... queries keys,... keys d_v -> ... queries d_v")

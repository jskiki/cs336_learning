"""作业 3.4.2：逐位置 SwiGLU 前馈网络的接口框架。

请完成各处 TODO，再接入 tests/adapters.py 中的 run_swiglu。
验证命令：在 assignment1-basics 下运行 uv run pytest -k test_swiglu。
"""

from __future__ import annotations

import torch
from torch import nn


def silu(x: torch.Tensor) -> torch.Tensor:
    """应用 SiLU 激活函数，输入和输出的形状相同。

    定义：SiLU(x) = x * sigmoid(x)。
    可以使用 torch.sigmoid，以保证数值稳定性。
    """
    # TODO 1：实现逐元素的 SiLU 激活，保留自动求导。
    original_dtype = x.dtype
    #其实这里可以进行一下类型的判断，当比float32的精度低时再进行转换，防止把高于float32的精度在降低了，但一般的大模型训练时float32的精度已经足够高了
    x = x.to(torch.float32)
    return (x*torch.sigmoid(x)).to(original_dtype)

class SwiGLU(nn.Module):
    """对每个位置独立应用同一个前馈网络。

    公式：FFN(x) = W2(SiLU(W1x) ⊙ W3x)。

    子模块命名及权重形状约定：
        w1.weight: (d_ff, d_model)。
        w2.weight: (d_model, d_ff)。
        w3.weight: (d_ff, d_model)。

    三个线性层均不使用偏置，可复用本包中已有的 Linear 模块。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """构造三个线性层。

        参数：
            d_model: 输入和输出的最后一维大小。
            d_ff: 中间层维度；为 None 时，选择接近 8/3 * d_model
                的 64 的倍数。显式传入时使用给定值，以匹配测试权重。
            device: 所有权重存储的设备。
            dtype: 所有权重的数据类型。

        完成标准：
            保存最终的 d_model 和 d_ff。
            将三个线性层注册为 self.w1、self.w2、self.w3。
            各层的权重形状符合类文档，且遵循 device 和 dtype。
        """
        super().__init__()
        self.d_model = d_model
        # TODO 2：确定并保存 self.d_ff，处理未显式指定 d_ff 的情况。
        # TODO 3：创建三个无偏置线性层，注意 w2 的输入和输出维度。
        raise NotImplementedError("请完成 SwiGLU 的维度配置和线性层构造")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 (..., d_model) 的输入变换为同样形状的输出。

        完成标准：
            按类文档中的公式组合线性变换、SiLU 和逐元素乘法。
            支持任意前导维度，例如 (batch_size, sequence_length)。
            每个位置独立计算，所有位置共享权重。
            保留输入和所有权重的自动求导。
        """
        # TODO 4：实现 SwiGLU 的前向计算。
        raise NotImplementedError("请实现 SwiGLU 前向计算")

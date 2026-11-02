"""作业 3.4.3：旋转位置嵌入（RoPE）的接口框架。

完成 TODO 后，再接入 tests/adapters.py 中的 run_rope。
验证命令：在 assignment1-basics 下运行 uv run pytest -k test_rope。
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """为查询或键向量应用 RoPE，不包含可学习参数。

    输入和输出形状均为 (..., seq_len, d_k)，支持任意数量的批次维度。
    d_k 应为正偶数；token_positions 指定实际位置，不一定从零连续递增。
    """
    #声明，防止Pylance的推断错误
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        """保存配置，并准备可复用的位置缓存。

        参数：
            theta: RoPE 的常数 Θ。
            d_k: 查询和键向量的最后一维大小。
            max_seq_len: 缓存支持的位置范围为 [0, max_seq_len)。
            device: 缓冲区所在的设备。

        完成标准：
            为支持的位置和元素对准备正弦、余弦缓存。
            每个缓存的形状为 (max_seq_len, d_k // 2)。
            使用 register_buffer(..., persistent=False) 注册缓存，
            使其随模块迁移设备，但不作为可学习参数或写入 state_dict。
            不构造完整的 d_k × d_k 旋转矩阵。
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # TODO 1：根据题目定义，计算位置和元素对对应的角度。
        #计算旋转频率
        inv_freq = self.theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)
        #计算每个位置的旋转角度（None用于新增一个长度为1的维度，方便后续广播）
        angles = torch.arange(max_seq_len, device=device)[:, None] * inv_freq[None, :]
        # TODO 2：计算并注册 cos_cached、sin_cached，遵循 device。
        """
        nn.Module 不仅提供计算接口，还会统一管理这个组件及其子组件中的参数、缓冲区和设备迁移。常称为模块
        register_buffer是nn.Module的一个方法，用于注册一个缓冲区（buffer），这个缓冲区不会被认为是模型的参数，也不会被优化器更新，但它会随着模型的迁移而迁移到不同的设备上。
        persistent=False表示这个缓冲区不会被保存到模型的状态字典中
        这两个的形状都为#(max_seq_len, d_k // 2)
        """
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """按指定位置旋转输入，返回形状相同的张量。

        参数：
            x: 形状为 (..., seq_len, d_k) 的查询或键张量。
            token_positions: 形状为 (..., seq_len) 的整数位置张量，
                其前导维度应能广播到 x 对应的前导维度。

        完成标准：
            使用 token_positions 索引缓存，支持非连续的位置编号。
            对最后一维中相邻的元素成对旋转，并保持原来的元素顺序。
            支持任意数量的批次维度，不逐个遍历 token。
            保持输出形状、设备和数据类型与 x 一致。
            保留 x 的自动求导，不原地修改输入。
        """
        #x的一个可能形状：(batch_size, seq_len, d_k) 批次大小、序列长度、向量维度
        # TODO 3：取出各 token 位置对应的缓存，并处理广播维度。
        #相当于把token_positions中的每个位置编号替换成缓存中的一行
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        even = x[..., 0::2] # 取出偶数位置的元素，最后一维的维度减半，因为是每两个一组
        odd = x[..., 1::2] # 取出奇数位置的元素
        # TODO 4：完成成对旋转，将结果恢复为输入形状。
        """
        stack在末尾新增一个维度，将每对结果放在一起,每对都是旋转结果
        flatten(-2)将最后两个维度展平，恢复到原来的形状
        """
        return torch.stack(
            (
                even * cos - odd * sin,
                even * sin + odd * cos,
            ),
            dim=-1
        ).flatten(-2).to(x.dtype) 

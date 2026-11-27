"""Linear、Embedding 和 RMSNorm 的基础层实现。"""

from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    """对输入的最后一个维度进行线性变换，保留所有前导维度。

    参数约定：使用名为 weight 的可训练参数，形状为
    (out_features, in_features)，按 W 的方向存储，不包含 bias。
    不得使用 nn.Linear 或 nn.functional.linear。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """构造线性层并初始化权重。

        参数：
            in_features: 输入最后一个维度的大小。
            out_features: 输出最后一个维度的大小。
            device: 权重存储的设备；为 None 时遵循 PyTorch 默认行为。
            dtype: 权重的数据类型；为 None 时遵循 PyTorch 默认行为。

        完成标准：
            weight 被注册为模块参数，形状、设备和数据类型符合约定。
            使用 torch.nn.init.trunc_normal_ 初始化权重：均值为 0，
            方差为 2 / (in_features + out_features)，截断范围为
            [-3σ, 3σ]，其中 σ 是标准差，不是方差。
        """
        super().__init__()
        # TODO 2：保存输入、输出维度，方便后续检查模块配置。
        # TODO 3：创建约定形状的权重，并以 nn.Parameter 注册为 self.weight。
        #         创建时应遵循传入的 device 和 dtype，不创建偏置参数。
        # TODO 4：按上述分布初始化权重，注意 std 与截断边界的含义。
        self.in_features = in_features
        self.out_features = out_features
        # nn.Parameter将张量注册为模型可训练的参数，torch.empty只创建不进行初始化，因此features必须有值
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2.0 / (in_features + out_features))
        # 函数名末尾的_表示原地修改，直接填写已有的self.weight，截断正态初始化
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将形状为 (..., in_features) 的输入变换为 (..., out_features)。

        完成标准：
            同时支持单个向量、批量输入和带序列维度的输入。
            所有前导维度保持不变，只对最后一个维度应用相同的权重。
            保留自动求导，使输入和权重能够参与反向传播。
        """
        # TODO 5：根据输入与所存权重的形状，实现无偏置线性变换。
        # TODO 6：检查返回值的最后一维及前导维度是否符合接口约定。
        return x @ self.weight.T


class Embedding(nn.Module):
    """将整数 token ID 映射到可训练的嵌入向量。

    参数约定：使用名为 weight 的可训练参数，形状为
    (num_embeddings, embedding_dim)，每一行对应一个 token。
    不得使用 nn.Embedding 或 nn.functional.embedding。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """构造嵌入层并初始化嵌入矩阵。

        参数：
            num_embeddings: 词表大小，即嵌入矩阵的行数。
            embedding_dim: 每个嵌入向量的维度，即 d_model。
            device: 权重存储的设备；为 None 时遵循 PyTorch 默认行为。
            dtype: 权重的数据类型；为 None 时遵循 PyTorch 默认行为。

        完成标准：
            weight 被注册为模块参数，最后一维是 embedding_dim。
            使用 torch.nn.init.trunc_normal_ 初始化权重：均值为 0，
            标准差为 1，截断范围为 [-3, 3]。
        """
        # TODO 1：调用 nn.Module 的父类构造函数。
        super().__init__()
        # TODO 2：保存词表大小和嵌入维度，方便后续检查模块配置。
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # TODO 3：创建约定形状的嵌入矩阵，并以 nn.Parameter 注册为 self.weight。
        #         权重应使用浮点类型，并遵循传入的 device 和 dtype。
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        # TODO 4：按上述截断正态分布初始化嵌入矩阵。
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """为每个 token ID 返回对应的嵌入向量。

        输入：整数索引张量，常见形状为 (batch_size, sequence_length)。
        输出：在输入形状末尾增加 embedding_dim 这一维。

        完成标准：
            支持测试适配器约定的任意前导维度，而非只支持二维输入。
            token ID 表示行索引，合法范围为 [0, num_embeddings)。
            输出数据类型与嵌入权重一致，并保留对权重的自动求导。
        """
        # TODO 5：根据 token ID 查询嵌入矩阵，保留原有索引维度与顺序。
        # TODO 6：检查输出形状是否等于输入形状追加一个嵌入维度。
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 可学习的缩放参数，对于输入x的最后一维的每个位置都有不同的缩放
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * inv_rms * self.weight).to(original_dtype)

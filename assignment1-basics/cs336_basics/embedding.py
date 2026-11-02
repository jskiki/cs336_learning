"""作业 3.3.3：Embedding 模块的接口框架。

请完成各处 TODO，再接入 tests/adapters.py 中的 run_embedding。
验证命令：uv run pytest -k test_embedding

权重矩阵与整个网络一块训练
"""

from __future__ import annotations

import torch
from torch import nn


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

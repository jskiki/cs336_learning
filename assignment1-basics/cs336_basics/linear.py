"""作业 3.3.2：无偏置 Linear 模块的接口框架。

请完成各处 TODO，再接入 tests/adapters.py 中的 run_linear。
验证命令：uv run pytest -k test_linear
"""

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
        #nn.Parameter将张量注册为模型可训练的参数，torch.empty只创建不进行初始化，因此features必须有值
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device = device,
                dtype = dtype
            )
        )

        std = math.sqrt(2.0/(in_features + out_features))
        #函数名末尾的_表示原地修改，直接填写已有的self.weight
        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a = -3.0*std,
            b = 3.0*std
        )


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

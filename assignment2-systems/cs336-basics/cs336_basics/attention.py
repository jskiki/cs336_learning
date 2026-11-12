"""作业 3.4.4–3.4.5：scaled dot-product attention 和 multi-head self-attention。"""

import torch
from torch import Tensor, nn
from math import sqrt
from einops import einsum,rearrange

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
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


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention，输入和输出均为 (..., seq_len, d_model)。

    使用自己的 Linear 和 scaled_dot_product_attention。
    四个无 bias 投影层的名称约定为 q_proj、k_proj、v_proj、output_proj，
    以便 adapter 加载对应的 weight。每个 weight 的形状均为 (d_model, d_model)。
    """

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    output_proj: Linear

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """保存配置并创建投影层。

        参数：
            d_model: 输入和输出的 embedding 维度，须能被 num_heads 整除。
            num_heads: attention head 数量，须为正数。
            rope: 可选 RoPE 模块；为 None 时不应用 RoPE。
            device: 投影层参数所在的设备。
            dtype: 投影层参数的数据类型。

        完成标准：
            设置每个 head 的维度 d_k = d_v = d_model // num_heads。
            创建上述四个 Linear 子模块，遵循 device 和 dtype。
            Q、K、V 各用一次矩阵乘法覆盖所有 head，共三次输入投影。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope

        # TODO 1：检查维度约定，保存每个 head 的维度。
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.d_k = d_model//num_heads
        self.d_v = self.d_k
        # TODO 2：初始化 q_proj、k_proj、v_proj、output_proj。
        self.q_proj = Linear(
            self.d_model,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            self.d_model,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            self.d_model,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            self.d_model,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """对序列应用 causal multi-head self-attention。

        参数：
            x: 浮点 tensor，形状为 (..., seq_len, d_model)。
            token_positions: 可选整数 tensor，形状为 (..., seq_len)。
                启用 RoPE 时用于指定位置；为 None 时使用从 0 开始的连续位置。
                位置应在传入 RoPE 模块的缓存范围内。

        完成标准：
            支持任意数量的前导 batch-like 维度，不逐个遍历 head 或 token。
            token i 只能关注 j <= i 的位置，包括自身。
            RoPE 只应用于 Q、K，各 head 使用相同的位置旋转规则。
            输出形状与 x 一致，并保留 autograd 计算图。
        """
        # TODO 3：使用投影层得到 Q、K、V。
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # TODO 4：拆分 head，得到 (..., num_heads, seq_len, d_k/d_v)。
        q_heads = rearrange(Q,"... seq_len (num_heads d_q) -> ... num_heads seq_len d_q",num_heads=self.num_heads)
        k_heads = rearrange(K,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads=self.num_heads)
        v_heads = rearrange(V,"... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",num_heads=self.num_heads)
        # TODO 5：如果启用了 RoPE，准备 token_positions 并处理 Q、K。
        #         head 作为 batch-like 维度，位置 tensor 应能广播到各 head。
        if self.rope is not None:
            positions = (
                torch.arange(q_heads.shape[-2], device=q_heads.device)
                if token_positions is None
                else token_positions.unsqueeze(-2) #unsqueeze(dim)在指定位置插入长度为1的维度，后再应用广播机制
            )
            q_heads = self.rope(q_heads, positions)
            k_heads = self.rope(k_heads, positions)
        # TODO 6：构造当前序列的 causal 布尔 mask，True 表示允许关注。
        #         mask 应与 x 位于同一 device。
        indices = torch.arange(x.shape[-2], device=x.device)
        #运用了广播机制，mask是一个二维张量
        mask = indices[:, None] >= indices[None, :]
        # TODO 7：调用已有的 scaled_dot_product_attention。
        mha = scaled_dot_product_attention(q_heads,k_heads,v_heads,mask)
        # TODO 8：合并 head，恢复 (..., seq_len, d_model)，再进行输出投影。
        output = rearrange(mha,"... heads seq_len d_v -> ... seq_len (heads d_v)")
        return self.output_proj(output)

"""作业 3.5：前置归一化 Transformer 块的接口框架。

完成 TODO 后，再接入 tests/adapters.py 中的 run_transformer_block。
验证命令：在 assignment1-basics 下运行 uv run pytest -k test_transformer_block。
"""

import torch
from torch import Tensor, nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rope import RotaryPositionalEmbedding



class TransformerBlock(nn.Module):
    """输入和输出形状均为 (..., seq_len, d_model)。

    使用已有的 RMSNorm、带 RoPE 的因果多头自注意力和 SwiGLU。
    子模块名称与测试提供的权重键保持一致：
        ln1.weight、ln2.weight。
        attn.q_proj.weight、attn.k_proj.weight、attn.v_proj.weight。
        attn.output_proj.weight。
        ffn.w1.weight、ffn.w2.weight、ffn.w3.weight。
    """

    ln1: RMSNorm
    attn: MultiHeadSelfAttention
    ln2: RMSNorm
    ffn: SwiGLU

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """保存配置并创建子模块。

        参数：
            d_model: 输入和输出的最后一维大小。
            num_heads: 注意力头数量，须能整除 d_model。
            d_ff: SwiGLU 的中间层维度。
            max_seq_len: RoPE 缓存支持的最大序列长度。
            theta: RoPE 的常数 Θ。
            eps: 两个 RMSNorm 使用的数值稳定项。
            device: 子模块参数和缓冲区所在的设备。
            dtype: 子模块参数的数据类型。

        完成标准：
            两个 RMSNorm 使用独立的可学习参数。
            RoPE 的维度匹配每个注意力头，而不是整个 d_model。
            注册类文档中约定的子模块名称，遵循 device 和 dtype。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # TODO 1：创建 self.ln1 和 self.ln2。
        self.ln1 = RMSNorm(self.d_model,eps=eps,device=device,dtype=dtype)
        self.ln2 = RMSNorm(self.d_model,eps=eps,device=device,dtype=dtype)
        # TODO 2：使用 rope.py 中的 RotaryPositionalEmbedding 准备 RoPE。
        self.rope = RotaryPositionalEmbedding(self.theta,self.d_model//self.num_heads,self.max_seq_len,device)
        # TODO 3：创建 self.attn，并传入 RoPE 模块。
        self.attn = MultiHeadSelfAttention(self.d_model,self.num_heads,self.rope,device,dtype)
        # TODO 4：创建 self.ffn。
        self.ffn = SwiGLU(self.d_model,self.d_ff,device,dtype)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """执行两个前置归一化子层，返回与输入同形状的张量。

        参数：
            x: 浮点张量，形状为 (..., seq_len, d_model)。
            token_positions: 可选整数位置张量，形状为 (..., seq_len)。
                为 None 时，沿用已有注意力模块的默认位置行为。

        完成标准：
            两个子层都在主要运算之前归一化，并各自使用残差连接。
            将 token_positions 传递给注意力模块。
            支持任意前导批次维度，不原地修改输入，并保留自动求导。
        """
        # TODO 5：完成注意力子层。
        attn_output = self.attn(self.ln1(x),token_positions)
        ffn_input = x + attn_output
        # TODO 6：完成前馈子层并返回结果。
        return ffn_input + self.ffn(self.ln2(ffn_input))

"""作业 3.5：完整 Transformer 语言模型的接口框架。

完成 TODO 后，再接入 tests/adapters.py 中的 run_transformer_lm。
验证命令：在 assignment1-basics 下运行 uv run pytest -k test_transformer_lm。
"""

import torch
from torch import Tensor, nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """将整数 token ID 序列映射为各位置的下一 token 预测分数。

    输入形状为 (..., seq_len)，输出形状为 (..., seq_len, vocab_size)。
    输出为未经 softmax 的 logits。

    子模块名称与测试提供的权重键保持一致：
        token_embeddings.weight: (vocab_size, d_model)。
        layers.0.*、layers.1.* 等：对应各个 TransformerBlock 的参数。
        ln_final.weight: (d_model,)。
        lm_head.weight: (vocab_size, d_model)。
    """

    token_embeddings: Embedding
    layers: nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """保存模型配置并创建子模块。

        参数：
            vocab_size: 词表大小。
            context_length: 最大上下文长度，同时用于配置各块的 RoPE 缓存。
            d_model: token 嵌入和各 Transformer 块的隐藏维度。
            num_layers: Transformer 块数量。
            num_heads: 每个块的注意力头数量。
            d_ff: 每个块中 SwiGLU 的中间层维度。
            rope_theta: 传给 TransformerBlock 的 theta 参数。
            eps: 各 RMSNorm 使用的数值稳定项。
            device: 子模块参数和缓冲区所在的设备。
            dtype: 子模块参数的数据类型。

        完成标准：
            复用本包中的 Embedding、TransformerBlock、RMSNorm 和 Linear。
            使用 nn.ModuleList 注册 num_layers 个独立的 TransformerBlock，
            各块拥有独立参数，而不是多次引用同一个块。
            将 context_length 对应到 TransformerBlock 的 max_seq_len。
            嵌入层与输出头分别拥有自己的权重，不共享参数。
            参数和缓冲区遵循传入的 device、dtype 及已有模块的约定。
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # TODO 1：创建 self.token_embeddings。
        self.token_embeddings = Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim =  self.d_model,
            device = device,
            dtype = dtype,                 
        )
        # TODO 2：创建 self.layers，包含指定数量的 TransformerBlock。
        #不能用普通的Python列表，使用nn.ModuleList，会将其中的块注册为子模块
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = self.d_model,
                    num_heads = self.num_heads,
                    d_ff = self.d_ff,
                    max_seq_len = self.context_length,
                    theta = self.rope_theta,
                    eps = eps,
                    device = device,
                    dtype = dtype,
                )
            for _ in range(self.num_layers)
            ]
        )
        # TODO 3：创建最终的归一化模块 self.ln_final。
        self.ln_final = RMSNorm(d_model=self.d_model,eps=eps,device=device,dtype=dtype)
        # TODO 4：创建从隐藏维度映射到词表维度的 self.lm_head。
        self.lm_head = Linear(
            in_features=self.d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """返回每个输入位置的下一 token 预测 logits。

        参数：
            token_ids: 整数张量，形状为 (..., seq_len)。
                token ID 的合法范围为 [0, vocab_size)。
                实际 seq_len 可以小于 context_length。
            token_positions: 可选整数位置张量，形状为 (..., seq_len)。
                传给各个 TransformerBlock；为 None 时使用其默认位置行为。
                位置编号须在 RoPE 缓存支持的范围内。

        完成标准：
            嵌入后及各 Transformer 块输出的形状为 (..., seq_len, d_model)。
            最终输出形状为 (..., seq_len, vocab_size)，不应用 softmax。
            支持短于最大上下文长度的输入，不补齐到 context_length。
            保留全部位置的输出以及参数的自动求导。
        """
        # TODO 5：查询 token 嵌入。
        hidden_states = self.token_embeddings(token_ids)
        # TODO 6：依次通过各个 Transformer 块，传递 token_positions。
        for layer in self.layers:
            hidden_states = layer(hidden_states,token_positions)
        # TODO 7：应用最终的 RMSNorm。
        output = self.ln_final(hidden_states)
        # TODO 8：通过语言模型输出头得到并返回 logits。
        #这里只是logits还没有经过最终的softmax装换成词表中各token的概率
        return self.lm_head(output)

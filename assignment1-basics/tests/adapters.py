from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

#from cs336_basics.train_bpe import train_bpe
from cs336_basics.fast_train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定 Linear 层的权重，计算批量输入经过线性变换后的结果。

    参数：
        d_in (int): 输入维度的大小。
        d_out (int): 输出维度的大小。
        weights (Float[Tensor, "d_out d_in"]): 使用的线性层权重。
        in_features (Float[Tensor, "... d_in"]): 待应用此函数的特征张量。

    返回值：
        Float[Tensor, "... d_out"]: 线性模块变换后的输出。
    """

    # TODO 1：从 cs336_basics.linear 导入你完成的 Linear 类。
    # TODO 2：根据 d_in、d_out 构造模块；设备和数据类型应与 weights 一致。
    # TODO 3：将给定 weights 加载到模块的 weight 参数中。
    #         可以使用 load_state_dict，注意状态字典的键与参数名称一致。
    #         测试应使用传入的权重，而不是模块随机初始化得到的权重。
    # TODO 4：将 in_features 交给模块计算，并返回输出张量。
    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层的权重，获取一批 token ID 对应的嵌入向量。

    参数：
        vocab_size (int): 词表中嵌入向量的数量。
        d_model (int): 嵌入维度的大小。
        weights (Float[Tensor, "vocab_size d_model"]): 从中提取嵌入向量的权重矩阵。
        token_ids (Int[Tensor, "..."]): 要从 Embedding 层查询的 token ID 集合。

    返回值：
        Float[Tensor, "... d_model"]: Embedding 层返回的一批嵌入向量。
    """

    # TODO 1：从 cs336_basics.embedding 导入你完成的 Embedding 类。
    # TODO 2：根据 vocab_size、d_model 构造模块；设备和数据类型应与 weights 一致。
    #         token_ids 是整数索引，不能用它的数据类型来创建浮点嵌入权重。
    # TODO 3：将给定 weights 加载到模块的 weight 参数中。
    #         可以使用 load_state_dict，测试应使用给定权重完成查询。
    # TODO 4：将 token_ids 交给模块查询，并返回嵌入张量。
    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 SwiGLU 网络的权重，返回你的实现使用这些权重计算得到的输出。

    参数：
        d_model (int): 前馈网络输入和输出的维度。
        d_ff (int): SwiGLU 内部升维投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): 保存的 W1 权重。
        w2_weight (Float[Tensor, "d_model d_ff"]): 保存的 W2 权重。
        w3_weight (Float[Tensor, "d_ff d_model"]): 保存的 W3 权重。
        in_features (Float[Tensor, "... d_model"]): 前馈层的输入嵌入。

    返回值：
        Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入。
    """
    # 示例：
    # 如果状态字典中的键与你的实现一致，可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 也可以手动赋值权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定键（K）、查询（Q）和值（V）张量，返回你的缩放点积注意力实现的输出。

    参数：
        Q (Float[Tensor, " ... queries d_k"]): 查询张量。
        K (Float[Tensor, " ... keys d_k"]): 键张量。
        V (Float[Tensor, " ... keys d_v"]): 值张量。
        mask (Bool[Tensor, " ... queries keys"] | None): 掩码张量。
    返回值：
        Float[Tensor, " ... queries d_v"]: 缩放点积注意力（SDPA）的输出。
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定朴素、未批处理的多头注意力实现中的键、查询和值投影权重，
    返回经过优化的批处理实现的输出。该实现应通过一次矩阵乘法，
    完成所有注意力头的键、查询和值投影。
    此函数不应使用 RoPE。
    参见 Vaswani 等人于 2017 年发表的论文第 3.2.2 节。

    参数：
        d_model (int): 前馈网络输入和输出的维度。
        num_heads (int): 多头注意力使用的注意力头数量。
        max_seq_len (int): 如果实现使用预缓存，则表示预缓存的最大序列长度。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重。
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重。
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重。
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重。
        in_features (Float[Tensor, "... sequence_length d_model"]): 传入你的实现进行计算的张量。

    返回值：
        Float[Tensor, " ... sequence_length d_model"]: 使用给定的 QKV 投影权重和输入特征，
        运行经过优化的批处理多头注意力实现后得到的输出张量。
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定朴素、未批处理的多头注意力实现中的键、查询和值投影权重，
    返回经过优化的批处理实现的输出。该实现应通过一次矩阵乘法，
    完成所有注意力头的键、查询和值投影。
    此版本的多头注意力（MHA）应包含 RoPE。
    此时，RoPE 的嵌入维度必须等于每个注意力头的嵌入维度（d_model // num_heads）。
    参见 Vaswani 等人于 2017 年发表的论文第 3.2.2 节。

    参数：
        d_model (int): 前馈网络输入和输出的维度。
        num_heads (int): 多头注意力使用的注意力头数量。
        max_seq_len (int): 如果实现使用预缓存，则表示预缓存的最大序列长度。
        theta (float): RoPE 参数。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重。
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重。
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重。
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重。
        in_features (Float[Tensor, "... sequence_length d_model"]): 传入你的实现进行计算的张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选张量，包含各 token 的位置。

    返回值：
        Float[Tensor, " ... sequence_length d_model"]: 使用给定的 QKV 投影权重和输入特征，
        运行经过优化的批处理多头注意力实现后得到的输出张量。
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定的输入张量应用旋转位置编码（RoPE）。

    参数：
        d_k (int): 查询或键张量的嵌入维度大小。
        theta (float): RoPE 参数。
        max_seq_len (int): 如果实现使用预缓存，则表示预缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 待应用 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 包含 token 位置的张量，形状为 (batch_size, sequence_length)。
    返回值：
        Float[Tensor, " ... sequence_length d_k"]: 对输入应用 RoPE 后得到的张量。
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定采用前置归一化的 Transformer 块的权重和输入特征，
    返回该 Transformer 块对输入特征进行计算后的输出。

    此函数应使用 RoPE。
    根据你的实现方式，可能只需将相关参数传给 TransformerBlock 的构造函数，
    也可能需要先初始化你自己的 RoPE 类，再将其实例传入。

    参数：
        d_model (int): Transformer 块输入的维度。
        num_heads (int): 多头注意力使用的注意力头数量。`d_model` 必须能被
            `num_heads` 整除。
        d_ff (int): 前馈网络内部层的维度。
        max_seq_len (int): 如果实现使用预缓存，则表示预缓存的最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]):
            参考实现的状态字典。
            此字典包含以下键：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影。
                形状为 (d_model, d_model)。
                各行按形状为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影。
                形状为 (d_model, d_model)。
                各行按形状为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影。
                形状为 (d_model, d_model)。
                各行按形状为 (num_heads, d_v) 的矩阵排列，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重。
                形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer 块中第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                前馈网络（FFN）中第一个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ffn.w2.weight`
                前馈网络中第二个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ffn.w3.weight`
                前馈网络中第三个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ln2.weight`
                Transformer 块中第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            传入你的实现进行计算的张量。

    返回值：
        Float[Tensor, "batch sequence_length d_model"] 使用 RoPE，
        对输入特征运行 Transformer 块后得到的输出张量。
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入索引，
    返回对这些输入索引执行前向传播后的输出。

    此函数应使用 RoPE。

    参数：
        vocab_size (int): 待预测的输出词表中不同词项的数量。
        context_length (int): 一次最多处理的 token 数量。
        d_model (int): 模型嵌入和各子层输出的维度。
        num_layers (int): 使用的 Transformer 层数。
        num_heads (int): 多头注意力使用的注意力头数量。`d_model` 必须能被
            `num_heads` 整除。
        d_ff (int): 前馈网络内部层的维度（第 3.3 节）。
        rope_theta (float): RoPE 的 $\\Theta$ 参数。
        weights (dict[str, Tensor]):
            参考实现的状态字典。{num_layers} 表示从 `0` 到
            `num_layers - 1` 之间的整数，即层索引。
            此字典包含以下键：
            - `token_embeddings.weight`
                token 嵌入矩阵，形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                各行按形状为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                各行按形状为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                各行按形状为 (num_heads, d_v) 的矩阵排列，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影的权重。
                形状为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                Transformer 块中第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                前馈网络中第一个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w2.weight`
                前馈网络中第二个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w3.weight`
                前馈网络中第三个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ln2.weight`
                Transformer 块中第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ln_final.weight`
                对最后一个 Transformer 块的输出应用 RMSNorm 时使用的仿射变换权重。
                形状为 (d_model, )。
            - `lm_head.weight`
                语言模型输出嵌入的权重。
                形状为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]) 输入语言模型的索引张量，形状为 (batch_size, sequence_length)，其中
            `sequence_length` 不超过 `context_length`。

    返回值：
        Float[Tensor, "batch_size sequence_length vocab_size"]: 包含每个 token 位置的
        下一个词预测结果的张量，预测结果为未归一化的分布。
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 RMSNorm 仿射变换的权重，
    返回对输入特征应用 RMSNorm 后的输出。

    参数：
        d_model (int): RMSNorm 输入的维度。
        eps: (float): 加入分母以提高数值稳定性的值。
        weights (Float[Tensor, "d_model"]): RMSNorm 的权重。
        in_features (Float[Tensor, "... d_model"]): 待应用 RMSNorm 的输入特征，前面可以有任意数量的维度。

    返回值：
        Float[Tensor,"... d_model"]: 对 `in_features` 应用 RMSNorm 后的输出张量，
        形状与 `in_features` 相同。
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定输入张量，返回对每个元素应用 SiLU 激活函数后的输出。

    参数：
        in_features(Float[Tensor, "..."]): 待应用 SiLU 的输入特征，形状可以任意。

    返回值：
        Float[Tensor,"..."]: 对每个元素应用 SiLU 后的输出张量，
        形状与 `in_features` 相同。
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（由整数组成的一维 NumPy 数组）、所需的批大小和上下文长度，
    从数据集中采样语言模型的输入序列及其对应标签。

    参数：
        dataset (np.array): 一维 NumPy 数组，包含数据集中的整数 token ID。
        batch_size (int): 希望采样的批大小。
        context_length (int): 每个采样样本所需的上下文长度。
        device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），指定采样得到的
            输入序列和标签应放置在哪个设备上。

    返回值：
        由两个形状均为 (batch_size, context_length) 的 torch.LongTensor 组成的元组。
        第一个元素是采样得到的输入序列，第二个元素是对应的语言模型训练标签。
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回沿指定维度 `dim` 应用 softmax 后的输出。

    参数：
        in_features (Float[Tensor, "..."]): 待应用 softmax 的输入特征，形状可以任意。
        dim (int): 在 `in_features` 中应用 softmax 的维度。

    返回值：
        Float[Tensor, "..."]: 沿指定维度 `dim` 进行 softmax 归一化后的张量，
        形状与 `in_features` 相同。
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入张量和目标张量，计算所有样本的平均交叉熵损失。

    参数：
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 表示
            第 i 个样本在第 j 个类别上的未归一化分数（logit）。
        targets (Int[Tensor, "batch_size"]): 包含正确类别索引的张量，形状为 (batch_size,)。
            每个值必须介于 0 和 `num_classes - 1` 之间。

    返回值：
        Float[Tensor, ""]: 所有样本的平均交叉熵损失。
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，裁剪它们的整体梯度，使其 L2 范数不超过 max_l2_norm。

    参数：
        parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
        max_l2_norm (float): 正数，表示允许的最大 L2 范数。

    应原地修改这些参数的梯度（parameter.grad）。
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    返回一个实现 AdamW 算法的 torch.optim.Optimizer 优化器类。
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定余弦学习率衰减调度（包含线性预热）的参数和迭代次数，
    返回该调度在指定迭代时的学习率。

    参数：
        it (int): 要查询学习率的迭代次数。
        max_learning_rate (float): alpha_max，余弦学习率调度（含预热）的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率调度（含预热）的最小学习率，
            也是最终学习率。
        warmup_iters (int): T_w，学习率线性预热的迭代次数。
        cosine_cycle_iters (int): T_c，余弦退火的迭代次数。

    返回值：
        指定调度在给定迭代次数下的学习率。
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代次数，将它们序列化保存到磁盘。

    参数：
        model (torch.nn.Module): 要序列化保存其状态的模型。
        optimizer (torch.optim.Optimizer): 要序列化保存其状态的优化器。
        iteration (int): 要序列化保存的数值，表示已经完成的训练迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 保存模型、优化器和迭代次数的目标路径或类文件对象。
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定序列化的检查点（路径或类文件对象），将其中保存的状态恢复到指定的模型和优化器。
    返回此前在检查点中保存的迭代次数。

    参数：
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点的路径或类文件对象。
        model (torch.nn.Module): 要恢复其状态的模型。
        optimizer (torch.optim.Optimizer): 要恢复其状态的优化器。
    返回值：
        int: 此前序列化保存的迭代次数。
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """给定词表、合并规则列表和特殊标记列表，
    返回使用这些词表、合并规则和特殊标记的 BPE 分词器。

    参数：
        vocab (dict[int, bytes]): 分词器词表，将整数（词表中的 token ID）
            映射到字节串（token 的字节内容）。
        merges (list[tuple[bytes, bytes]]): BPE 合并规则，每一项都是字节串元组 (<token1>, <token2>)，
            表示 <token1> 与 <token2> 被合并。
            合并规则按产生的先后顺序排列。
        special_tokens (list[str] | None): 分词器使用的特殊标记字符串列表。
            这些字符串始终作为单个 token 保留，不会被拆分成多个 token。

    返回值：
        使用给定词表、合并规则和特殊标记的 BPE 分词器。
    """
    return Tokenizer(
        vocab = vocab,
        merges = merges,
        special_tokens = special_tokens,
    )


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料的路径，训练 BPE 分词器，并输出其词表和合并规则。

    参数：
        input_path (str | os.PathLike): BPE 分词器训练数据的路径。
        vocab_size (int): 分词器词表中的词项总数，包括特殊标记。
        special_tokens (list[str]): 要加入分词器词表的特殊标记字符串列表。
            这些字符串始终作为单个 token 保留，不会被拆分成多个 token。
            如果这些特殊标记出现在 `input_path` 指定的文件中，
            则按其他字符串的方式处理。

    返回值：
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练得到的分词器词表，将整数（词表中的 token ID）
                映射到字节串（token 的字节内容）。
            merges:
                BPE 合并规则，每一项都是字节串元组 (<token1>, <token2>)，
                表示 <token1> 与 <token2> 被合并。
                合并规则按产生的先后顺序排列。
    """
    return train_bpe(
    input_path=str(input_path),
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)

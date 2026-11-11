"""作业 6：文本生成框架，核心计算由你完成。

复用 TransformerLM、Tokenizer 和 cs336_basics.softmax 中已有的接口。
建议按温度缩放、top-p 采样、token 生成、文本生成的顺序完成 TODO。
本框架先支持单条 prompt，不包含批量生成或 KV cache。
"""

import math
import torch
from torch import Tensor

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.softmax import softmax


def temperature_softmax(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """将下一 token 的 logits 转换为经过温度缩放的概率分布。

    参数：
        logits: 有限浮点张量，形状为 (vocab_size,)。
        temperature: 有限且大于 0 的温度值，1.0 表示不调整温度。
            本接口不接受 0；如需贪心解码，可以另行扩展。

    返回值：
        形状为 (vocab_size,) 的概率张量，设备与输入一致。

    完成标准：
        复用 cs336_basics.softmax.softmax 的数值稳定实现。
        概率非负、有限，且总和约为 1；不原地修改 logits。
        temperature=1.0 时与普通 softmax 一致。
        非法 temperature 抛出 ValueError。
    """
    # TODO 1：检查温度参数。
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("温度必须是有限且大于 0 的数值")
    # TODO 2：按题目公式完成温度缩放与 
    if temperature == 1.0:
        return softmax(logits, dim=-1)
    scaled_logits = (
        logits - logits.amax(dim=-1, keepdim=True)
    ) / temperature
    #整个tensor都进行了计算，dim只是决定怎么分组
    return softmax(scaled_logits, dim=-1)


def top_p_sampling(probabilities: Tensor, top_p: float = 1.0) -> Tensor:
    """从给定概率分布进行核采样，返回一个 token ID。

    参数：
        probabilities: 形状为 (vocab_size,) 的有效概率分布。
        top_p: 累计概率阈值，满足 0 < top_p <= 1。

    返回值：
        标量 torch.long 张量，值为原始词表中的 token ID。
        返回张量与 probabilities 位于同一设备。

    完成标准：
        保留按概率降序排列后累计概率首次达到或超过阈值的最小前缀。
        包含使累计概率达到或超过阈值的那个 token，且至少保留一个候选。
        top_p=1.0 时使用完整分布。
        按截断后重新归一化的概率采样，返回原词表 ID。
        不原地修改 probabilities；非法 top_p 抛出 ValueError。
    """
    # TODO 3：检查阈值参数。
    if not 0<top_p<=1:
        raise ValueError("top_p的值必须在(0,1]之间")
    # TODO 4：按题目定义确定核采样的候选集合。
    # 低精度输入转为 float32，减少累计概率的舍入误差。
    weights = (
        probabilities.float()
        if probabilities.dtype in (torch.float16, torch.bfloat16)
        else probabilities
    )
    # 不截断时跳过排序。
    if top_p == 1.0:
        return torch.multinomial(weights, num_samples=1).squeeze(0)
    #从大到小排序，sorted_indices排序后对用的下标
    sorted_probs, sorted_indices = torch.sort(weights, descending=True)
    #cumsum是沿指定维度计算累积和，如sorted_probs = [0.4, 0.3, 0.2, 0.1]则cumulative_probs = [0.4, 0.7, 0.9, 1.0]
    cumulative_probs = sorted_probs.cumsum(dim=0)
    # 前一个位置已达到阈值时，当前位置才被移除。
    # 因此保留首次达到阈值的 token，并保证至少保留一个。
    #masked_fill_第一个参数要求的是布尔的Tensor,cumulative_probs[:-1] >= top_p逐元素比较，最终得到的也是bool
    sorted_probs[1:].masked_fill_(cumulative_probs[:-1] >= top_p, 0)
    # TODO 5：完成采样，并确认返回的是原始词表中的 ID。
    # multinomial 接受未归一化的非负权重，无需额外归一化。（即按权重随机抽取一个），sampled_index是排序后概率在数组中的位置
    sampled_index = torch.multinomial(
        sorted_probs, num_samples=1
    ).squeeze(0)
    return sorted_indices[sampled_index]


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    """根据单条 token 序列生成续写。

    参数：
        model: 已加载权重的语言模型，使用其 context_length 限制上下文。
        prompt_ids: 非空 torch.long 张量，形状为 (prompt_length,)。
            token ID 位于 [0, model.vocab_size)，设备与模型一致。
        max_new_tokens: 最多新增的 token 数量，必须是非负整数。
        eos_token_id: 序列结束标记的 ID；None 表示仅按长度停止。
        temperature: 有限且大于 0 的温度值。
        top_p: 核采样阈值，满足 0 < top_p <= 1。

    返回值：
        包含完整 prompt 和新增 token 的一维 torch.long 张量。
        若生成 EOS，返回结果包含该 EOS；设备与 prompt_ids 一致。

    完成标准：
        生成期间使用评估模式，退出时恢复模型原有的训练或评估状态。
        不记录梯度、不修改模型参数、不原地修改 prompt_ids。
        每次前向的输入长度不超过 model.context_length。
        超长时仅限制模型看到的上下文，返回结果仍保留完整历史。
        使用最后一个输入位置的 logits 预测下一个 token。
        复用 temperature_softmax 和 top_p_sampling。
        新生成 EOS 后立即停止；prompt 中已有 EOS 不触发提前停止。
        max_new_tokens=0 时返回与 prompt 相同的序列内容。
        空 prompt、非法参数或不符合约定的输入抛出 ValueError。
    """
    if max_new_tokens == 0:
        return prompt_ids.clone()
    #numel():返回张量中的元素总数
    length = prompt_ids.numel()
    #new_empty根据现有张量的dtype和device创建一个指定形状的新张量，但不初始化元素值
    output = prompt_ids.new_empty(length+max_new_tokens)
    output[:length].copy_(prompt_ids)
    # TODO 7：完成自回归生成，并处理上下文长度与停止条件。
    #保存下模型状态，后面要转成eval模式，was_training是布尔值
    was_training = model.training
    try:
        model.eval()

        for _ in range(max_new_tokens):
            #防止超出最大长度，截取最后的context_length个token
            #unsqueeze(0),增加批次维度：(seq_len,) → (1, seq_len)
            #模型输出形状是 (1, seq_len, vocab_size)，[0, -1] 取第一个样本、最后一个输入位置的 logits，得到形状为 (vocab_size,) 的向量，用来预测下一个 token
            logits = model(
                output[
                    max(0,length - model.context_length):length
                ].unsqueeze(0)
            )[0,-1]
            next_token = top_p_sampling(
                temperature_softmax(logits, temperature),
                top_p,
            )

            output[length] = next_token
            length += 1
            #item()将只有一个元素的Tensor转成Python数值
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            #预留的output的空间可能用不完，只返回有效的token
        return output[:length]
    # TODO 8：确保正常返回或发生异常时，均恢复模型原有模式。
    finally:
        model.train(was_training)


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """接收文本 prompt，返回包含 prompt 的完整生成文本。

    参数：
        model: 已加载权重的语言模型。
        tokenizer: 与模型训练时使用的词表、合并规则和特殊标记一致的分词器。
        prompt: 编码后非空的提示文本。
        max_new_tokens、temperature、top_p: 与 generate 的约定相同。

    完成标准：
        复用 Tokenizer.encode、generate 和 Tokenizer.decode。
        根据模型所在设备准备 token 张量。
        从 tokenizer.token_to_id 获取 <|endoftext|> 的真实 ID，不硬编码。
        缺少结束标记时抛出 ValueError，且不临时扩充模型词表。
        返回文本保留生成的 <|endoftext|>；展示层可以自行去除。
    """
    # TODO 9：完成文本与 token 接口之间的衔接。
    eos_token_id = tokenizer.token_to_id.get(b"<|endoftext|>")
    if eos_token_id is None:
        raise ValueError("分词器词表中缺少 <|endoftext|>")
    #next(...)：取出模型的第一个参数。.device：读取这个参数所在的设备。
    prompt_ids = torch.tensor(tokenizer.encode(prompt),dtype=torch.long,device=next(model.parameters()).device,)
    generate_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p
    )

    return tokenizer.decode(generate_ids.tolist())

"""纯 PyTorch FlashAttention-2：分块前向与重计算反向。"""

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


Q_TILE_SIZE = 16
K_TILE_SIZE = 16


def _tiled_forward(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> tuple[Tensor, Tensor]:
    """计算并返回注意力输出 O 和逐 query 的 logsumexp L。

    Q 的形状为 (..., n_queries, d)，K 和 V 为 (..., n_keys, d)。
    O 的形状与 Q 相同，L 的形状为 (..., n_queries)。
    """
    if K.shape[-2] == 0:
        raise ValueError("key 序列不能为空")

    acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
    output = torch.empty_like(Q)
    logsumexp = Q.new_empty(Q.shape[:-1], dtype=acc_dtype)
    scale = Q.shape[-1] ** -0.5
    positions = torch.arange(max(Q.shape[-2], K.shape[-2]), device=Q.device) if is_causal else None

    # 每个 K 块和同一段 token 的 V 块一起遍历。
    key_tiles = K.to(acc_dtype).transpose(-2, -1).split(K_TILE_SIZE, dim=-1)
    value_tiles = V.to(acc_dtype).split(K_TILE_SIZE, dim=-2)
    key_value_tiles = tuple(zip(key_tiles, value_tiles, strict=True))

    for start in range(0, Q.shape[-2], Q_TILE_SIZE):
        query_slice = slice(start, start + Q_TILE_SIZE)
        query = Q[..., query_slice, :].to(acc_dtype)
        query_positions = positions[start : start + query.shape[-2]] if positions is not None else None

        # 逐 query 保存已读分数的最大值、指数和及加权 V 累加值。
        row_max = query.new_full((*query.shape[:-1], 1), -torch.inf)
        row_sum = torch.zeros_like(row_max)
        weighted_values = torch.zeros_like(query)
        for tile, (keys_transposed, values) in enumerate(key_value_tiles):
            key_start = tile * K_TILE_SIZE
            if is_causal and key_start >= start + query.shape[-2]:
                break

            scores = (query @ keys_transposed).mul_(scale)
            if query_positions is not None and positions is not None:
                key_positions = positions[key_start : key_start + values.shape[-2]]
                scores.masked_fill_(query_positions[:, None] < key_positions[None, :], -torch.inf)
            new_max = torch.maximum(row_max, scores.amax(dim=-1, keepdim=True))

            # 最大值更新后，把之前的累加结果转换到同一尺度。
            rescale = (row_max - new_max).exp_()
            weights = scores.sub_(new_max).exp_()

            row_sum.mul_(rescale).add_(weights.sum(dim=-1, keepdim=True))
            weighted_values.mul_(rescale).add_(weights @ values)
            row_max = new_max
        # 所有可见 key 块处理完后，再归一化并写回。
        output[..., query_slice, :] = weighted_values.div_(row_sum)
        logsumexp[..., query_slice] = (row_max + row_sum.log_()).squeeze(-1)

    return output, logsumexp

#fullgraph=True要求一次把函数捕获为一张计算图，遇到图中断会报错
# @torch.compile(fullgraph=True)
# def _recompute_backward(
#     Q: Tensor, K: Tensor, V: Tensor, output: Tensor, grad_output: Tensor, logsumexp: Tensor, is_causal: bool,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     #is_causal决定是否使用因果掩码
#     """从保存的 logsumexp 重算注意力权重，返回 Q、K、V 的梯度。"""
#     acc_dtype = logsumexp.dtype
#     query, keys, values = Q.to(acc_dtype), K.to(acc_dtype), V.to(acc_dtype)
#     grad_output = grad_output.to(acc_dtype)
#     scale = Q.shape[-1] ** -0.5

#     #重计算注意力权重
#     scores = (query @ keys.transpose(-2, -1)) * scale
#     if is_causal:
#         query_positions = torch.arange(Q.shape[-2], device=Q.device)
#         key_positions = torch.arange(K.shape[-2], device=K.device)
#         #query_positions[:, None] 的形状是 (Nq, 1)，`key_positions[None, :]` 是 (1, Nk)。比较时广播成 (Nq, Nk)，当 \(i<j\) 时结果为 True。`masked_fill` 把这些位置的分数替换成 -inf，其余位置保留原来的分数。
#         scores = scores.masked_fill(query_positions[:, None] < key_positions[None, :], -torch.inf)
#     probabilities = (scores - logsumexp.unsqueeze(-1)).exp()

#     # softmax 反向中的逐行内积 D = sum(O * dO)。
#     row_dot = (output.to(acc_dtype) * grad_output).sum(dim=-1, keepdim=True)
#     grad_probabilities = grad_output @ values.transpose(-2, -1)
#     grad_scores = probabilities * (grad_probabilities - row_dot)
#     grad_query = (grad_scores @ keys) * scale
#     grad_keys = (grad_scores.transpose(-2, -1) @ query) * scale 
#     grad_values = probabilities.transpose(-2, -1) @ grad_output
#     return grad_query.to(Q.dtype), grad_keys.to(K.dtype), grad_values.to(V.dtype)

#分块的反向
@torch.no_grad()
def _recompute_backward(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    output: Tensor,
    grad_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """分块重算注意力权重，并累加 Q、K、V 的梯度。"""
    acc_dtype = logsumexp.dtype
    query, keys, values = Q.to(acc_dtype), K.to(acc_dtype), V.to(acc_dtype)
    grad_output = grad_output.to(acc_dtype)

    n_queries, n_keys = Q.shape[-2], K.shape[-2]
    scale = Q.shape[-1] ** -0.5

    # 每个 query 的 D = O · dO，在所有 key 块之间共享。
    row_dot = (output.to(acc_dtype) * grad_output).sum(dim=-1, keepdim=True)

    grad_query = torch.zeros_like(query)
    grad_keys = torch.empty_like(keys)
    grad_values = torch.empty_like(values)

    #取 max(n_queries, n_keys) 是为了让同一份编号覆盖两条可能不同长度的序列，交叉注意力中Q，可能不等长
    positions = (
        torch.arange(max(n_queries, n_keys), device=Q.device)
        if is_causal
        else None
    )

    for key_start in range(0, n_keys, K_TILE_SIZE):
        key_slice = slice(key_start, key_start + K_TILE_SIZE)
        key_tile = keys[..., key_slice, :]
        value_tile = values[..., key_slice, :]
        keys_t = key_tile.transpose(-2, -1)
        values_t = value_tile.transpose(-2, -1)

        grad_key_tile = torch.zeros_like(key_tile)
        grad_value_tile = torch.zeros_like(value_tile)

        # 因果模式下，跳过整块都看不到当前 key 块的早期 query。找到能看到当前key的第一个query块
        first_query = key_start // Q_TILE_SIZE * Q_TILE_SIZE if is_causal else 0

        for query_start in range(first_query, n_queries, Q_TILE_SIZE):
            query_slice = slice(query_start, query_start + Q_TILE_SIZE)
            query_tile = query[..., query_slice, :]
            grad_output_tile = grad_output[..., query_slice, :]

            scores = (query_tile @ keys_t).mul_(scale)
            if positions is not None:
                #先把位置范围限制到各自的实际序列长度，再切出当前块
                query_positions = positions[:n_queries][query_slice]
                key_positions = positions[:n_keys][key_slice]
                scores.masked_fill_(
                    query_positions[:, None] < key_positions[None, :],
                    -torch.inf,
                )

            # L 来自前向，已经包含每行所有可见 key 的归一化信息。
            probabilities = scores.sub_(logsumexp[..., query_slice, None]).exp_()
            grad_probabilities = grad_output_tile @ values_t
            grad_scores = probabilities * (
                grad_probabilities - row_dot[..., query_slice, :]
            )

            grad_query[..., query_slice, :].add_(grad_scores @ key_tile, alpha=scale)
            grad_key_tile.add_(grad_scores.transpose(-2, -1) @ query_tile, alpha=scale)
            grad_value_tile.add_(probabilities.transpose(-2, -1) @ grad_output_tile)

        grad_keys[..., key_slice, :] = grad_key_tile
        grad_values[..., key_slice, :] = grad_value_tile

    return grad_query.to(Q.dtype), grad_keys.to(K.dtype), grad_values.to(V.dtype)


@once_differentiable
def _backward_once(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
    logsumexp, Q, K, V, output = ctx.saved_tensors
    grads = _recompute_backward(Q, K, V, output, grad_output, logsumexp, ctx.is_causal)
    # is_causal 可省略，按 .apply 实际接收的参数数量返回梯度。最后的 None 对应布尔参数 is_causal
    return (*grads, None)[: len(ctx.needs_input_grad)]


class FlashAttention2PyTorch(torch.autograd.Function):
    """通过 .apply(Q, K, V, is_causal) 调用，支持可选的因果掩码。"""

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        output, logsumexp = _tiled_forward(Q, K, V, is_causal)
        ctx.save_for_backward(logsumexp, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        (grad_output,) = grad_outputs
        return _backward_once(ctx, grad_output)

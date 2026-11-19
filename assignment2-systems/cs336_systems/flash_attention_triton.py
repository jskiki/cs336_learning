"""作业 4.2.2(b–c)：Triton FlashAttention-2 前向。"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    query_start = query_tile_index * Q_TILE_SIZE

    q_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    k_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    v_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    o_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    l_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 每个程序实例只加载一次 Q，并用 FP32 累加在线 softmax 状态。
    query = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
    row_max = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    row_sum = tl.zeros((Q_TILE_SIZE,), tl.float32)
    accumulator = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    query_positions = query_start + tl.arange(0, Q_TILE_SIZE)
    key_offsets = tl.arange(0, K_TILE_SIZE)
    key_limit = N_KEYS
    if is_causal:
        # 跳过完全位于当前 query 块之后的 K/V 块。
        key_limit = tl.minimum(N_KEYS, query_start + Q_TILE_SIZE)

    for tile in range(tl.cdiv(key_limit, K_TILE_SIZE)):
        keys = tl.load(k_ptr, boundary_check=(0,), padding_option="zero")
        values = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")
        scores = tl.dot(query, tl.trans(keys), input_precision="tf32x3") * scale

        key_positions = tile * K_TILE_SIZE + key_offsets
        valid = key_positions[None, :] < N_KEYS
        if is_causal:
            valid = valid & (query_positions[:, None] >= key_positions[None, :])
        scores = tl.where(valid, scores, -float("inf"))

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        rescale = tl.exp(row_max - new_max)
        weights = tl.exp(scores - new_max[:, None])
        row_sum = row_sum * rescale + tl.sum(weights, axis=1)
        accumulator = tl.dot(
            weights.to(values.dtype), values,
            acc=accumulator * rescale[:, None],
            input_precision="tf32x3",
        )
        row_max = new_max

        k_ptr = k_ptr.advance((K_TILE_SIZE, 0))
        v_ptr = v_ptr.advance((K_TILE_SIZE, 0))

    output = accumulator / row_sum[:, None]
    logsumexp = row_max + tl.log(row_sum)
    tl.store(o_ptr, output.to(query.dtype), boundary_check=(0,))
    tl.store(l_ptr, logsumexp, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    """通过 .apply 接收 (batch, sequence, d) 的 CUDA 张量。"""

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        if K.shape != (batch_size, n_keys, d) or V.shape != K.shape:
            raise ValueError("Q、K、V 的 batch 和特征维必须相同，K 和 V 的序列长度必须相同")
        if n_queries == 0 or n_keys == 0 or d < 16 or triton.next_power_of_2(d) != d:
            raise ValueError("序列不能为空，特征维必须是不小于 16 的 2 的幂")
        if not Q.is_cuda or Q.device != K.device or Q.device != V.device:
            raise ValueError("Q、K、V 必须位于同一 CUDA 设备")
        if Q.dtype != K.dtype or Q.dtype != V.dtype or Q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Q、K、V 必须具有相同的 FP16、BF16 或 FP32 dtype")

        # 连续布局与块指针的 order=(1, 0) 对应。
        query, keys, values = Q.contiguous(), K.contiguous(), V.contiguous()
        output = torch.empty_like(query)
        logsumexp = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

        q_tile_size = 32
        k_tile_size = 32
        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
        with torch.cuda.device(Q.device):
            flash_fwd_kernel[grid](
                query, keys, values, output, logsumexp,
                *query.stride(), *keys.stride(), *values.stride(),
                *output.stride(), *logsumexp.stride(),
                n_queries, n_keys, d ** -0.5,
                D=d,
                Q_TILE_SIZE=q_tile_size,
                K_TILE_SIZE=k_tile_size,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )

        ctx.save_for_backward(logsumexp, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        raise NotImplementedError("FlashAttention-2 反向传播将在后续小题实现")

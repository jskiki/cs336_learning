"""Triton FlashAttention-2：大 tile autotune、因果无掩码分段、Block Pointer / TMA 与两遍无原子反向。

可选环境变量（用于 B200 线下调优或固定最优配置以压缩冷启动编译时间）：
- CS336_FLASH_AUTOTUNE=1: 启用更大范围的 autotune 搜索空间。
- CS336_FLASH_USE_TMA=0: 关闭 Hopper/Blackwell TMA，回退到 tl.make_block_ptr。
- CS336_FLASH_FWD_CFG / CS336_FLASH_DQ_CFG / CS336_FLASH_DKV_CFG:
  以 "Q_TILE,K_TILE,num_warps,num_stages"（如 "128,64,4,3"）固定单组配置。
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

# 内部 softmax 统一用 2 为底的指数（tl.exp2 对应单条硬件 EX2 指令）：
# exp(s) = 2^(s * log2(e))；写回自然对数 logsumexp 时再乘 ln(2) 还原。
# Triton 3.x 要求内核中读取的全局常量必须用 tl.constexpr(...) 实例化。
_RCP_LN2: float = 1.4426950408889634
_RCP_LN2_TL = tl.constexpr(1.4426950408889634)
_LN2_TL = tl.constexpr(0.6931471805599453)
_TMA_ALLOCATOR_SET: bool = False


def _make_configs(env_name: str, defaults: list[tuple[int, int, int, int]], full_tiles: tuple[tuple[int, int], ...], full_stages: tuple[int, ...]) -> list[triton.Config]:
    """按优先级返回 autotune 配置：环境变量固定单组 > 线下全搜 > 精简默认 4 组。"""
    if raw := os.environ.get(env_name):
        parts = [int(x.strip()) for x in raw.split(",")]
        if len(parts) != 4:
            raise ValueError(f"{env_name} 格式应为 'Q_TILE,K_TILE,num_warps,num_stages'，收到：{raw!r}")
        q, k, w, s = parts
        return [triton.Config({"Q_TILE_SIZE": q, "K_TILE_SIZE": k}, num_warps=w, num_stages=s)]
    if os.environ.get("CS336_FLASH_AUTOTUNE") == "1":
        return [triton.Config({"Q_TILE_SIZE": q, "K_TILE_SIZE": k}, num_warps=w, num_stages=s) for q, k in full_tiles for w in (4, 8) for s in full_stages]
    return [triton.Config({"Q_TILE_SIZE": q, "K_TILE_SIZE": k}, num_warps=w, num_stages=s) for q, k, w, s in defaults]


_FWD_CONFIGS = _make_configs(
    "CS336_FLASH_FWD_CFG",
    [(128, 64, 4, 3), (128, 128, 8, 3), (128, 64, 8, 4), (64, 64, 4, 2)],
    full_tiles=((64, 64), (128, 64), (128, 128), (64, 128)),
    full_stages=(2, 3, 4),
)
_BWD_DQ_CONFIGS = _make_configs(
    "CS336_FLASH_DQ_CFG",
    [(128, 64, 8, 2), (64, 64, 4, 3), (64, 64, 4, 2), (32, 32, 4, 2)],
    full_tiles=((64, 64), (128, 64), (128, 128), (32, 32)),
    full_stages=(2, 3),
)
_BWD_DKV_CONFIGS = _make_configs(
    "CS336_FLASH_DKV_CFG",
    [(64, 128, 8, 2), (64, 64, 4, 3), (64, 64, 4, 2), (32, 32, 4, 2)],
    full_tiles=((64, 64), (64, 128), (128, 128), (32, 32)),
    full_stages=(2, 3),
)


def _ensure_tma_allocator() -> None:
    """Triton 设备端 make_tensor_descriptor 需要注册全局显存分配器，用于存放 TMA 描述符。"""
    global _TMA_ALLOCATOR_SET
    if not _TMA_ALLOCATOR_SET:
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        _TMA_ALLOCATOR_SET = True


def _can_use_tma(*tensors: Tensor, d: int) -> bool:
    """检查是否满足 SM90+/B200 TMA 条件：API 可用、架构 >= Hopper、首地址与行字节步长 16 字节对齐。"""
    if os.environ.get("CS336_FLASH_USE_TMA", "1") == "0":
        return False
    if not (hasattr(tl, "make_tensor_descriptor") and hasattr(triton, "set_allocator")):
        return False
    if torch.cuda.get_device_capability(tensors[0].device)[0] < 9 or d < 16:
        return False
    return all(t.stride(-1) == 1 and (t.stride(-2) * t.element_size()) % 16 == 0 and t.data_ptr() % 16 == 0 for t in tensors)


def _dot_precision(dtype: torch.dtype) -> str:
    # BF16/FP16 传 "ieee" 直接走原生 Tensor Core MMA + FP32 累加；FP32 保持 tf32x3 以满足高精度单测。
    return "tf32x3" if dtype == torch.float32 else "ieee"


@triton.jit
def _flash_fwd_step(
    query, keys, values, row_max, row_sum, accumulator, scale_log2,
    query_positions, key_positions, N_KEYS,
    MASK_BOUNDARY: tl.constexpr, MASK_CAUSAL: tl.constexpr, INPUT_PRECISION: tl.constexpr,
):
    """在线 softmax 单块更新；当 MASK_* 全为 False 时，编译期直接裁掉所有比较与 tl.where。"""
    scores = tl.dot(query, tl.trans(keys), input_precision=INPUT_PRECISION) * scale_log2
    if MASK_BOUNDARY and MASK_CAUSAL:
        scores = tl.where((key_positions[None, :] < N_KEYS) & (query_positions[:, None] >= key_positions[None, :]), scores, -float("inf"))
    elif MASK_BOUNDARY:
        scores = tl.where(key_positions[None, :] < N_KEYS, scores, -float("inf"))
    elif MASK_CAUSAL:
        scores = tl.where(query_positions[:, None] >= key_positions[None, :], scores, -float("inf"))

    new_max = tl.maximum(row_max, tl.max(scores, axis=1))
    rescale = tl.exp2(row_max - new_max)
    weights = tl.exp2(scores - new_max[:, None])
    row_sum = row_sum * rescale + tl.sum(weights, axis=1)
    accumulator = tl.dot(weights.to(values.dtype), values, acc=accumulator * rescale[:, None], input_precision=INPUT_PRECISION)
    return new_max, row_sum, accumulator


@triton.autotune(configs=_FWD_CONFIGS, key=["N_QUERIES", "N_KEYS", "D", "is_causal", "USE_TMA", "INPUT_PRECISION"])
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd, stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd, stride_ob, stride_oq, stride_od, stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale_log2,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr, USE_TMA: tl.constexpr, INPUT_PRECISION: tl.constexpr,
):
    query_start = tl.program_id(0) * Q_TILE_SIZE
    # 用 int64 计算 batch 偏移，防止大 batch * 长序列场景下 32 位整型溢出。
    batch_index = tl.program_id(1).to(tl.int64)

    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(Q_ptr + batch_index * stride_qb, shape=[N_QUERIES, D], strides=[stride_qq, 1], block_shape=[Q_TILE_SIZE, D])
        k_desc = tl.make_tensor_descriptor(K_ptr + batch_index * stride_kb, shape=[N_KEYS, D], strides=[stride_kk, 1], block_shape=[K_TILE_SIZE, D])
        v_desc = tl.make_tensor_descriptor(V_ptr + batch_index * stride_vb, shape=[N_KEYS, D], strides=[stride_vk, 1], block_shape=[K_TILE_SIZE, D])
        o_desc = tl.make_tensor_descriptor(O_ptr + batch_index * stride_ob, shape=[N_QUERIES, D], strides=[stride_oq, 1], block_shape=[Q_TILE_SIZE, D])
        query = q_desc.load([query_start, 0])
    else:
        q_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb, (N_QUERIES, D), (stride_qq, stride_qd), (query_start, 0), (Q_TILE_SIZE, D), (1, 0))
        k_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb, (N_KEYS, D), (stride_kk, stride_kd), (0, 0), (K_TILE_SIZE, D), (1, 0))
        v_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb, (N_KEYS, D), (stride_vk, stride_vd), (0, 0), (K_TILE_SIZE, D), (1, 0))
        o_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob, (N_QUERIES, D), (stride_oq, stride_od), (query_start, 0), (Q_TILE_SIZE, D), (1, 0))
        query = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")

    l_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb, (N_QUERIES,), (stride_lq,), (query_start,), (Q_TILE_SIZE,), (0,))

    row_max = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    row_sum = tl.zeros((Q_TILE_SIZE,), tl.float32)
    accumulator = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    query_positions = query_start + tl.arange(0, Q_TILE_SIZE)
    key_offsets = tl.arange(0, K_TILE_SIZE)

    # 因果分块边界：
    # - key_limit：最大可能被当前 Q 块看到的 key 上界（超过 query_start + Q_TILE_SIZE 的块全被遮挡，直接跳过）。
    # - full_limit：完全不需要掩码的 key 上界。当 key_tile 满足 k_max <= query_start（即 k_end <= query_start + 1）
    #   且不越过 N_KEYS 时，块内所有 (q, k) 必然满足 q >= k 且 k < N_KEYS，可进入零掩码快速循环。
    if is_causal:
        key_limit = tl.minimum(N_KEYS, query_start + Q_TILE_SIZE)
        full_limit = tl.minimum(N_KEYS, query_start + 1)
    else:
        key_limit = N_KEYS
        full_limit = N_KEYS

    full_tiles = full_limit // K_TILE_SIZE
    total_tiles = tl.cdiv(key_limit, K_TILE_SIZE)

    # 阶段 1：完全在界内且无因果遮挡的 K/V 块（热路径，无边界检查、无索引比较）。
    for tile in range(0, full_tiles):
        if USE_TMA:
            keys = k_desc.load([tile * K_TILE_SIZE, 0])
            values = v_desc.load([tile * K_TILE_SIZE, 0])
        else:
            keys = tl.load(k_ptr)
            values = tl.load(v_ptr)
            k_ptr = k_ptr.advance((K_TILE_SIZE, 0))
            v_ptr = v_ptr.advance((K_TILE_SIZE, 0))
        row_max, row_sum, accumulator = _flash_fwd_step(
            query, keys, values, row_max, row_sum, accumulator, scale_log2,
            query_positions, key_offsets, N_KEYS, MASK_BOUNDARY=False, MASK_CAUSAL=False, INPUT_PRECISION=INPUT_PRECISION,
        )

    # 阶段 2：对角线块与末尾非整除边界块（仅 1~2 个块，在此执行掩码）。
    for tile in range(full_tiles, total_tiles):
        key_start = tile * K_TILE_SIZE
        if USE_TMA:
            keys = k_desc.load([key_start, 0])
            values = v_desc.load([key_start, 0])
        else:
            keys = tl.load(k_ptr, boundary_check=(0,), padding_option="zero")
            values = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")
            k_ptr = k_ptr.advance((K_TILE_SIZE, 0))
            v_ptr = v_ptr.advance((K_TILE_SIZE, 0))
        row_max, row_sum, accumulator = _flash_fwd_step(
            query, keys, values, row_max, row_sum, accumulator, scale_log2,
            query_positions, key_start + key_offsets, N_KEYS, MASK_BOUNDARY=True, MASK_CAUSAL=is_causal, INPUT_PRECISION=INPUT_PRECISION,
        )

    # TMA store 要求张量 dtype 与目标指针严格一致，先显式转换回 query.dtype。
    output = (accumulator / row_sum[:, None]).to(query.dtype)
    logsumexp = (row_max + tl.log2(row_sum)) * _LN2_TL
    if USE_TMA:
        o_desc.store([query_start, 0], output)
    else:
        tl.store(o_ptr, output, boundary_check=(0,))
    tl.store(l_ptr, logsumexp, boundary_check=(0,))


@triton.jit
def _flash_bwd_row_dot(O_ptr, DO_ptr, Row_dot_ptr, N_QUERIES, D: tl.constexpr, Q_TILE_SIZE: tl.constexpr):
    """预计算 softmax 反向逐行内积 D_i = sum_d (O_id * dO_id)；输入均为连续 (B, N_QUERIES, D)。"""
    query_start = tl.program_id(0) * Q_TILE_SIZE
    batch_index = tl.program_id(1).to(tl.int64)
    q_base = batch_index * N_QUERIES * D

    o_ptr = tl.make_block_ptr(O_ptr + q_base, (N_QUERIES, D), (D, 1), (query_start, 0), (Q_TILE_SIZE, D), (1, 0))
    do_ptr = tl.make_block_ptr(DO_ptr + q_base, (N_QUERIES, D), (D, 1), (query_start, 0), (Q_TILE_SIZE, D), (1, 0))
    d_ptr = tl.make_block_ptr(Row_dot_ptr + batch_index * N_QUERIES, (N_QUERIES,), (1,), (query_start,), (Q_TILE_SIZE,), (0,))

    output = tl.load(o_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    grad_output = tl.load(do_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    tl.store(d_ptr, tl.sum(output * grad_output, axis=1), boundary_check=(0,))


@triton.jit
def _flash_bwd_tile(
    query, keys, values, grad_output, logsumexp_log2, row_dot, scale_log2,
    query_positions, key_positions, N_QUERIES, N_KEYS,
    MASK_Q_BOUNDARY: tl.constexpr, MASK_K_BOUNDARY: tl.constexpr, MASK_CAUSAL: tl.constexpr, INPUT_PRECISION: tl.constexpr,
):
    """从保存的 L 重算块内概率 P 与分数梯度 dS = P * (dO @ V^T - D)。"""
    scores = tl.dot(query, tl.trans(keys), input_precision=INPUT_PRECISION) * scale_log2
    if MASK_Q_BOUNDARY and MASK_CAUSAL:
        scores = tl.where((query_positions[:, None] < N_QUERIES) & (query_positions[:, None] >= key_positions[None, :]), scores - logsumexp_log2[:, None], -float("inf"))
    elif MASK_K_BOUNDARY and MASK_CAUSAL:
        scores = tl.where((key_positions[None, :] < N_KEYS) & (query_positions[:, None] >= key_positions[None, :]), scores - logsumexp_log2[:, None], -float("inf"))
    elif MASK_Q_BOUNDARY:
        scores = tl.where(query_positions[:, None] < N_QUERIES, scores - logsumexp_log2[:, None], -float("inf"))
    elif MASK_K_BOUNDARY:
        scores = tl.where(key_positions[None, :] < N_KEYS, scores - logsumexp_log2[:, None], -float("inf"))
    else:
        scores = scores - logsumexp_log2[:, None]

    probabilities = tl.exp2(scores)
    grad_probabilities = tl.dot(grad_output, tl.trans(values), input_precision=INPUT_PRECISION)
    grad_scores = probabilities * (grad_probabilities - row_dot[:, None])
    return probabilities, grad_scores


@triton.jit
def _flash_bwd_dkv_step(
    query, keys, values, grad_output, logsumexp_log2, row_dot, grad_keys, grad_values,
    scale_log2, query_positions, key_positions, N_QUERIES, N_KEYS,
    MASK_Q_BOUNDARY: tl.constexpr, MASK_CAUSAL: tl.constexpr, INPUT_PRECISION: tl.constexpr,
):
    """累加当前 query 块对 dK、dV 的贡献；三个分段循环共用此计算体。"""
    probabilities, grad_scores = _flash_bwd_tile(
        query, keys, values, grad_output, logsumexp_log2, row_dot, scale_log2,
        query_positions, key_positions, N_QUERIES, N_KEYS,
        MASK_Q_BOUNDARY=MASK_Q_BOUNDARY, MASK_K_BOUNDARY=False, MASK_CAUSAL=MASK_CAUSAL, INPUT_PRECISION=INPUT_PRECISION,
    )
    grad_keys = tl.dot(tl.trans(grad_scores).to(query.dtype), query, acc=grad_keys, input_precision=INPUT_PRECISION)
    grad_values = tl.dot(tl.trans(probabilities).to(grad_output.dtype), grad_output, acc=grad_values, input_precision=INPUT_PRECISION)
    return grad_keys, grad_values


@triton.autotune(configs=_BWD_DQ_CONFIGS, key=["N_QUERIES", "N_KEYS", "D", "is_causal", "USE_TMA", "INPUT_PRECISION"])
@triton.jit
def _flash_bwd_dq(
    Q_ptr, K_ptr, V_ptr, DO_ptr, L_ptr, Row_dot_ptr, DQ_ptr,
    N_QUERIES, N_KEYS, scale, scale_log2,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr, USE_TMA: tl.constexpr, INPUT_PRECISION: tl.constexpr,
):
    """第一遍反向：按 Q 块并行计算 dQ，各 block 独占输出行，写回无需原子操作。"""
    query_start = tl.program_id(0) * Q_TILE_SIZE
    batch_index = tl.program_id(1).to(tl.int64)
    q_base = batch_index * N_QUERIES * D
    k_base = batch_index * N_KEYS * D
    v_row_base = batch_index * N_QUERIES

    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(Q_ptr + q_base, shape=[N_QUERIES, D], strides=[D, 1], block_shape=[Q_TILE_SIZE, D])
        do_desc = tl.make_tensor_descriptor(DO_ptr + q_base, shape=[N_QUERIES, D], strides=[D, 1], block_shape=[Q_TILE_SIZE, D])
        k_desc = tl.make_tensor_descriptor(K_ptr + k_base, shape=[N_KEYS, D], strides=[D, 1], block_shape=[K_TILE_SIZE, D])
        v_desc = tl.make_tensor_descriptor(V_ptr + k_base, shape=[N_KEYS, D], strides=[D, 1], block_shape=[K_TILE_SIZE, D])
        dq_desc = tl.make_tensor_descriptor(DQ_ptr + q_base, shape=[N_QUERIES, D], strides=[D, 1], block_shape=[Q_TILE_SIZE, D])
        query = q_desc.load([query_start, 0])
        grad_output = do_desc.load([query_start, 0])
    else:
        q_ptr = tl.make_block_ptr(Q_ptr + q_base, (N_QUERIES, D), (D, 1), (query_start, 0), (Q_TILE_SIZE, D), (1, 0))
        do_ptr = tl.make_block_ptr(DO_ptr + q_base, (N_QUERIES, D), (D, 1), (query_start, 0), (Q_TILE_SIZE, D), (1, 0))
        k_ptr = tl.make_block_ptr(K_ptr + k_base, (N_KEYS, D), (D, 1), (0, 0), (K_TILE_SIZE, D), (1, 0))
        v_ptr = tl.make_block_ptr(V_ptr + k_base, (N_KEYS, D), (D, 1), (0, 0), (K_TILE_SIZE, D), (1, 0))
        dq_ptr = tl.make_block_ptr(DQ_ptr + q_base, (N_QUERIES, D), (D, 1), (query_start, 0), (Q_TILE_SIZE, D), (1, 0))
        query = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
        grad_output = tl.load(do_ptr, boundary_check=(0,), padding_option="zero")

    l_ptr = tl.make_block_ptr(L_ptr + v_row_base, (N_QUERIES,), (1,), (query_start,), (Q_TILE_SIZE,), (0,))
    d_ptr = tl.make_block_ptr(Row_dot_ptr + v_row_base, (N_QUERIES,), (1,), (query_start,), (Q_TILE_SIZE,), (0,))
    logsumexp_log2 = tl.load(l_ptr, boundary_check=(0,), padding_option="zero") * _RCP_LN2_TL
    row_dot = tl.load(d_ptr, boundary_check=(0,), padding_option="zero")
    grad_query = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    query_positions = query_start + tl.arange(0, Q_TILE_SIZE)
    key_offsets = tl.arange(0, K_TILE_SIZE)
    key_limit = tl.minimum(N_KEYS, query_start + Q_TILE_SIZE) if is_causal else N_KEYS
    full_limit = tl.minimum(N_KEYS, query_start + 1) if is_causal else N_KEYS
    full_tiles = full_limit // K_TILE_SIZE
    total_tiles = tl.cdiv(key_limit, K_TILE_SIZE)

    # 阶段 1：完全可见的内部 K/V 块（dQ 各行独立，越界 Q 行在最终 store 时被裁掉，故无需任何掩码）。
    for tile in range(0, full_tiles):
        if USE_TMA:
            keys = k_desc.load([tile * K_TILE_SIZE, 0])
            values = v_desc.load([tile * K_TILE_SIZE, 0])
        else:
            keys = tl.load(k_ptr)
            values = tl.load(v_ptr)
            k_ptr = k_ptr.advance((K_TILE_SIZE, 0))
            v_ptr = v_ptr.advance((K_TILE_SIZE, 0))
        _, grad_scores = _flash_bwd_tile(
            query, keys, values, grad_output, logsumexp_log2, row_dot, scale_log2,
            query_positions, key_offsets, N_QUERIES, N_KEYS,
            MASK_Q_BOUNDARY=False, MASK_K_BOUNDARY=False, MASK_CAUSAL=False, INPUT_PRECISION=INPUT_PRECISION,
        )
        grad_query = tl.dot(grad_scores.to(keys.dtype), keys, acc=grad_query, input_precision=INPUT_PRECISION)

    # 阶段 2：对角线块与尾部边界块。
    for tile in range(full_tiles, total_tiles):
        key_start = tile * K_TILE_SIZE
        if USE_TMA:
            keys = k_desc.load([key_start, 0])
            values = v_desc.load([key_start, 0])
        else:
            keys = tl.load(k_ptr, boundary_check=(0,), padding_option="zero")
            values = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")
            k_ptr = k_ptr.advance((K_TILE_SIZE, 0))
            v_ptr = v_ptr.advance((K_TILE_SIZE, 0))
        _, grad_scores = _flash_bwd_tile(
            query, keys, values, grad_output, logsumexp_log2, row_dot, scale_log2,
            query_positions, key_start + key_offsets, N_QUERIES, N_KEYS,
            MASK_Q_BOUNDARY=False, MASK_K_BOUNDARY=True, MASK_CAUSAL=is_causal, INPUT_PRECISION=INPUT_PRECISION,
        )
        grad_query = tl.dot(grad_scores.to(keys.dtype), keys, acc=grad_query, input_precision=INPUT_PRECISION)

    out_dq = (grad_query * scale).to(query.dtype)
    if USE_TMA:
        dq_desc.store([query_start, 0], out_dq)
    else:
        tl.store(dq_ptr, out_dq, boundary_check=(0,))


@triton.autotune(configs=_BWD_DKV_CONFIGS, key=["N_QUERIES", "N_KEYS", "D", "is_causal", "USE_TMA", "INPUT_PRECISION"])
@triton.jit
def _flash_bwd_dkv(
    Q_ptr, K_ptr, V_ptr, DO_ptr, L_ptr, Row_dot_ptr, DK_ptr, DV_ptr,
    N_QUERIES, N_KEYS, scale, scale_log2,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr, USE_TMA: tl.constexpr, INPUT_PRECISION: tl.constexpr,
):
    """第二遍反向：按 K/V 块并行计算 dK、dV，同样无需原子操作。"""
    key_start = tl.program_id(0) * K_TILE_SIZE
    batch_index = tl.program_id(1).to(tl.int64)
    q_base = batch_index * N_QUERIES * D
    k_base = batch_index * N_KEYS * D
    v_row_base = batch_index * N_QUERIES

    # 将 Q 块范围 [first_q_tile, total_q_tiles) 无重叠、无遗漏地切成连续三段：
    # 1. [first_q_tile, diag_end_tile): 因果对角线块（首个能看到 key_start 的块，到最后一个含 q < key_max 的块）。
    #    其中当前 K 块最大位置为 key_start + K_TILE_SIZE - 1，只有当 q_start >= key_max 才彻底脱离对角线，
    #    因此 diag_end_tile = cdiv(key_start + K_TILE_SIZE - 1, Q_TILE_SIZE)。
    # 2. [diag_end_tile, full_q_tiles): 完全在界内且全部满足 q >= k 的内部块（零掩码热路径）。
    # 3. [max(diag_end_tile, full_q_tiles), total_q_tiles): 仅当 N_QUERIES 非 Q_TILE_SIZE 整数倍时的末尾块。
    total_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    full_q_tiles = N_QUERIES // Q_TILE_SIZE
    if is_causal:
        first_q_tile = tl.minimum(key_start // Q_TILE_SIZE, total_q_tiles)
        diag_end_tile = tl.minimum(tl.cdiv(key_start + K_TILE_SIZE - 1, Q_TILE_SIZE), total_q_tiles)
    else:
        first_q_tile = 0
        diag_end_tile = 0
    first_q_offset = first_q_tile * Q_TILE_SIZE

    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(Q_ptr + q_base, shape=[N_QUERIES, D], strides=[D, 1], block_shape=[Q_TILE_SIZE, D])
        do_desc = tl.make_tensor_descriptor(DO_ptr + q_base, shape=[N_QUERIES, D], strides=[D, 1], block_shape=[Q_TILE_SIZE, D])
        k_desc = tl.make_tensor_descriptor(K_ptr + k_base, shape=[N_KEYS, D], strides=[D, 1], block_shape=[K_TILE_SIZE, D])
        v_desc = tl.make_tensor_descriptor(V_ptr + k_base, shape=[N_KEYS, D], strides=[D, 1], block_shape=[K_TILE_SIZE, D])
        dk_desc = tl.make_tensor_descriptor(DK_ptr + k_base, shape=[N_KEYS, D], strides=[D, 1], block_shape=[K_TILE_SIZE, D])
        dv_desc = tl.make_tensor_descriptor(DV_ptr + k_base, shape=[N_KEYS, D], strides=[D, 1], block_shape=[K_TILE_SIZE, D])
        keys = k_desc.load([key_start, 0])
        values = v_desc.load([key_start, 0])
    else:
        q_ptr = tl.make_block_ptr(Q_ptr + q_base, (N_QUERIES, D), (D, 1), (first_q_offset, 0), (Q_TILE_SIZE, D), (1, 0))
        do_ptr = tl.make_block_ptr(DO_ptr + q_base, (N_QUERIES, D), (D, 1), (first_q_offset, 0), (Q_TILE_SIZE, D), (1, 0))
        k_ptr = tl.make_block_ptr(K_ptr + k_base, (N_KEYS, D), (D, 1), (key_start, 0), (K_TILE_SIZE, D), (1, 0))
        v_ptr = tl.make_block_ptr(V_ptr + k_base, (N_KEYS, D), (D, 1), (key_start, 0), (K_TILE_SIZE, D), (1, 0))
        dk_ptr = tl.make_block_ptr(DK_ptr + k_base, (N_KEYS, D), (D, 1), (key_start, 0), (K_TILE_SIZE, D), (1, 0))
        dv_ptr = tl.make_block_ptr(DV_ptr + k_base, (N_KEYS, D), (D, 1), (key_start, 0), (K_TILE_SIZE, D), (1, 0))
        keys = tl.load(k_ptr, boundary_check=(0,), padding_option="zero")
        values = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")

    l_ptr = tl.make_block_ptr(L_ptr + v_row_base, (N_QUERIES,), (1,), (first_q_offset,), (Q_TILE_SIZE,), (0,))
    d_ptr = tl.make_block_ptr(Row_dot_ptr + v_row_base, (N_QUERIES,), (1,), (first_q_offset,), (Q_TILE_SIZE,), (0,))
    grad_keys = tl.zeros((K_TILE_SIZE, D), tl.float32)
    grad_values = tl.zeros((K_TILE_SIZE, D), tl.float32)
    key_positions = key_start + tl.arange(0, K_TILE_SIZE)
    query_offsets = tl.arange(0, Q_TILE_SIZE)

    # 阶段 1（仅因果模式）：对角线块，同时屏蔽越界 Q 与 q < k。
    if is_causal:
        for tile in range(first_q_tile, diag_end_tile):
            query_start = tile * Q_TILE_SIZE
            if USE_TMA:
                query = q_desc.load([query_start, 0])
                grad_output = do_desc.load([query_start, 0])
            else:
                query = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
                grad_output = tl.load(do_ptr, boundary_check=(0,), padding_option="zero")
                q_ptr = q_ptr.advance((Q_TILE_SIZE, 0))
                do_ptr = do_ptr.advance((Q_TILE_SIZE, 0))
            logsumexp_log2 = tl.load(l_ptr, boundary_check=(0,), padding_option="zero") * _RCP_LN2_TL
            row_dot = tl.load(d_ptr, boundary_check=(0,), padding_option="zero")
            l_ptr = l_ptr.advance((Q_TILE_SIZE,))
            d_ptr = d_ptr.advance((Q_TILE_SIZE,))
            grad_keys, grad_values = _flash_bwd_dkv_step(
                query, keys, values, grad_output, logsumexp_log2, row_dot, grad_keys, grad_values,
                scale_log2, query_start + query_offsets, key_positions, N_QUERIES, N_KEYS,
                MASK_Q_BOUNDARY=True, MASK_CAUSAL=True, INPUT_PRECISION=INPUT_PRECISION,
            )

    # 阶段 2：完全可见的内部 Q 块（热路径，零掩码、无边界检查加载）。
    for tile in range(diag_end_tile, full_q_tiles):
        if USE_TMA:
            query = q_desc.load([tile * Q_TILE_SIZE, 0])
            grad_output = do_desc.load([tile * Q_TILE_SIZE, 0])
        else:
            query = tl.load(q_ptr)
            grad_output = tl.load(do_ptr)
            q_ptr = q_ptr.advance((Q_TILE_SIZE, 0))
            do_ptr = do_ptr.advance((Q_TILE_SIZE, 0))
        logsumexp_log2 = tl.load(l_ptr) * _RCP_LN2_TL
        row_dot = tl.load(d_ptr)
        l_ptr = l_ptr.advance((Q_TILE_SIZE,))
        d_ptr = d_ptr.advance((Q_TILE_SIZE,))
        grad_keys, grad_values = _flash_bwd_dkv_step(
            query, keys, values, grad_output, logsumexp_log2, row_dot, grad_keys, grad_values,
            scale_log2, query_offsets, key_positions, N_QUERIES, N_KEYS,
            MASK_Q_BOUNDARY=False, MASK_CAUSAL=False, INPUT_PRECISION=INPUT_PRECISION,
        )

    # 阶段 3：非整除序列长度的最后一个尾部 Q 块（已越过对角线，只需检查 q < N_QUERIES）。
    for tile in range(tl.maximum(diag_end_tile, full_q_tiles), total_q_tiles):
        query_start = tile * Q_TILE_SIZE
        if USE_TMA:
            query = q_desc.load([query_start, 0])
            grad_output = do_desc.load([query_start, 0])
        else:
            query = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
            grad_output = tl.load(do_ptr, boundary_check=(0,), padding_option="zero")
            q_ptr = q_ptr.advance((Q_TILE_SIZE, 0))
            do_ptr = do_ptr.advance((Q_TILE_SIZE, 0))
        logsumexp_log2 = tl.load(l_ptr, boundary_check=(0,), padding_option="zero") * _RCP_LN2_TL
        row_dot = tl.load(d_ptr, boundary_check=(0,), padding_option="zero")
        l_ptr = l_ptr.advance((Q_TILE_SIZE,))
        d_ptr = d_ptr.advance((Q_TILE_SIZE,))
        grad_keys, grad_values = _flash_bwd_dkv_step(
            query, keys, values, grad_output, logsumexp_log2, row_dot, grad_keys, grad_values,
            scale_log2, query_start + query_offsets, key_positions, N_QUERIES, N_KEYS,
            MASK_Q_BOUNDARY=True, MASK_CAUSAL=False, INPUT_PRECISION=INPUT_PRECISION,
        )

    out_dk = (grad_keys * scale).to(keys.dtype)
    out_dv = grad_values.to(values.dtype)
    if USE_TMA:
        dk_desc.store([key_start, 0], out_dk)
        dv_desc.store([key_start, 0], out_dv)
    else:
        tl.store(dk_ptr, out_dk, boundary_check=(0,))
        tl.store(dv_ptr, out_dv, boundary_check=(0,))


@once_differentiable
def _backward_once(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
    logsumexp, query, keys, values, output = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    batch_size, n_queries, d = query.shape
    n_keys = keys.shape[1]

    row_dot = torch.empty_like(logsumexp)
    grad_query = torch.empty_like(query)
    grad_keys = torch.empty_like(keys)
    grad_values = torch.empty_like(values)

    scale = d**-0.5
    scale_log2 = scale * _RCP_LN2
    input_precision = _dot_precision(query.dtype)
    use_tma = _can_use_tma(query, keys, values, grad_output, grad_query, grad_keys, grad_values, d=d)
    if use_tma:
        _ensure_tma_allocator()

    row_dot_tile = 128 if n_queries >= 128 else 32
    dq_grid = lambda meta: (triton.cdiv(n_queries, meta["Q_TILE_SIZE"]), batch_size)  # noqa: E731
    dkv_grid = lambda meta: (triton.cdiv(n_keys, meta["K_TILE_SIZE"]), batch_size)  # noqa: E731

    with torch.cuda.device(query.device):
        _flash_bwd_row_dot[(triton.cdiv(n_queries, row_dot_tile), batch_size)](output, grad_output, row_dot, n_queries, D=d, Q_TILE_SIZE=row_dot_tile, num_warps=4)
        _flash_bwd_dq[dq_grid](
            query, keys, values, grad_output, logsumexp, row_dot, grad_query,
            n_queries, n_keys, scale, scale_log2, D=d, is_causal=ctx.is_causal, USE_TMA=use_tma, INPUT_PRECISION=input_precision,
        )
        _flash_bwd_dkv[dkv_grid](
            query, keys, values, grad_output, logsumexp, row_dot, grad_keys, grad_values,
            n_queries, n_keys, scale, scale_log2, D=d, is_causal=ctx.is_causal, USE_TMA=use_tma, INPUT_PRECISION=input_precision,
        )

    return (grad_query, grad_keys, grad_values, None)[: len(ctx.needs_input_grad)]


class FlashAttention2Triton(torch.autograd.Function):
    """通过 .apply(Q, K, V, is_causal) 接收 (batch, sequence, d) 的 CUDA 张量。"""

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

        # 保证连续布局，同时满足 block_ptr 的 order=(1, 0) 与 TMA 2D 描述符步长要求。
        query, keys, values = Q.contiguous(), K.contiguous(), V.contiguous()
        output = torch.empty_like(query)
        logsumexp = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

        scale_log2 = (d**-0.5) * _RCP_LN2
        input_precision = _dot_precision(query.dtype)
        use_tma = _can_use_tma(query, keys, values, output, d=d)
        if use_tma:
            _ensure_tma_allocator()

        grid = lambda meta: (triton.cdiv(n_queries, meta["Q_TILE_SIZE"]), batch_size)  # noqa: E731
        with torch.cuda.device(Q.device):
            flash_fwd_kernel[grid](
                query, keys, values, output, logsumexp,
                *query.stride(), *keys.stride(), *values.stride(), *output.stride(), *logsumexp.stride(),
                n_queries, n_keys, scale_log2, D=d, is_causal=is_causal, USE_TMA=use_tma, INPUT_PRECISION=input_precision,
            )

        ctx.save_for_backward(logsumexp, query, keys, values, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        (grad_output,) = grad_outputs
        return _backward_once(ctx, grad_output)


FlashAttention2TritonOpt = FlashAttention2Triton

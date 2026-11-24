"""全分片数据并行。各 rank 需以相同顺序执行参与通信的层。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.autograd.graph import saved_tensors_hooks

from cs336_basics.model import Embedding, Linear

#记录本地分片和原始形状（整个参数而非分片上的参数）
@dataclass(frozen=True)
class _Shard:
    parameter: nn.Parameter
    shape: torch.Size
    numel: int

#封装一次all-gathre，这样FSDP可以提前发起后面层的收集，等到了那一层再执行wait
@dataclass
class _Gather:
    tensor: Tensor
    work: dist.Work | None

    def wait(self) -> Tensor:
        if self.work is not None:
            self.work.wait()
        return self.tensor

#记录反向所需的权重视图
@dataclass(frozen=True)
class _SavedWeight:
    shard: _Shard
    shape: torch.Size
    stride: tuple[int, ...]
    offset: int


class _UnshardWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter: nn.Parameter, full_weight: Tensor, fsdp: FSDP) -> Tensor:
        ctx.parameter = parameter
        ctx.fsdp = fsdp
        return full_weight

    @staticmethod
    def backward(ctx, *gradients: Tensor) -> tuple[None, None, None]:
        # 通信可以与剩余反向计算重叠，等到 finish 再把平均梯度交给优化器。
        (gradient,) = gradients
        ctx.fsdp._reduce_shard_gradient(ctx.parameter, gradient)
        return None, None, None


class FSDP(nn.Module):
    """分片 Linear 和 Embedding 的参数，其余参数保持复制。

    在创建优化器之前包装模型。反向传播后、更新参数前调用
    finish_gradient_synchronization，以便优化器读取本 rank 的平均梯度。
    """

    @torch.no_grad()
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._rank = dist.get_rank() if self.world_size > 1 else 0
        self._uses_nccl = self.world_size > 1 and dist.get_backend() == "nccl"

        self._layers: dict[nn.Module, dict[str, nn.Parameter]] = {
            layer: {name: param for name, param in layer.named_parameters(recurse=False) if name in ("weight", "bias")}
            for layer in module.modules()
            if isinstance(layer, (nn.Linear, nn.Embedding, Linear, Embedding))#需要分片的参数
        }
        #收集所有需要分片的参数
        sharded = {param for params in self._layers.values() for param in params.values()}
        self._shards: dict[nn.Parameter, _Shard] = {}
        #不分片的完整权重，如归一化层
        self._replicated: list[nn.Parameter] = []
        for param in module.parameters():
            #先全部广播，再从中切分该rank负责的部分
            if self.world_size > 1 and param.numel():
                dist.broadcast(param, src=0)
            if param not in sharded:
                self._replicated.append(param)
                continue

            self._shards[param] = _Shard(param, param.shape, param.numel())
            #参数分片
            shard_size = (param.numel() + self.world_size - 1) // self.world_size
            start = self._rank * shard_size
            #从起点开始能复制到的真实元素
            length = max(0, min(shard_size, param.numel() - start))
            #如果指定compute_dtype就用FP32存参数，计算时转换，没指定就沿用原来的参数类型
            shard = param.new_zeros(shard_size, dtype=torch.float32 if compute_dtype is not None else param.dtype)
            if length:
                #如果不能平分，里面会有0做填充
                shard[:length].copy_(param.detach().reshape(-1).narrow(0, start, length))
            #让原来的 Parameter 对象改为持有这块更小的存储，应为adamw会使用这个，只更新本rank参数即可
            param.data = shard

        #保存上一次前向中，分片层实际执行的顺序
        self._forward_order: list[nn.Module] = []
        self._current_order: list[nn.Module] = []
        self._prefetched: tuple[nn.Module, dict[str, _Gather]] | None = None
        self._gradient_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._pending_gradients: list[tuple[nn.Parameter, Tensor, dist.Work | None]] = []
        for layer, params in self._layers.items():
            self._wrap_layer(layer, params)

    @torch.no_grad()
    def _gather(self, shard: _Shard, dtype: torch.dtype | None) -> _Gather:
        local = shard.parameter.detach().to(dtype=dtype)
        work = None
        if self.world_size > 1 and local.numel():
            full = local.new_empty(local.numel() * self.world_size)
            work = dist.all_gather_into_tensor(full, local, async_op=True)
        else:
            full = local
        return _Gather(full[: shard.numel].view(shard.shape), work)

    def _gather_layer(self, layer: nn.Module) -> dict[str, _Gather]:
        return {name: self._gather(self._shards[param], self.compute_dtype) for name, param in self._layers[layer].items()}

    def _discard_prefetch(self) -> None:
        if self._prefetched is not None:
            for gather in self._prefetched[1].values():
                gather.wait()
            self._prefetched = None

    def _prepare_layer(self, layer: nn.Module) -> dict[str, _Gather]:
        position = len(self._current_order)
        self._current_order.append(layer)
        if self._prefetched is not None and self._prefetched[0] is layer:
            gathers = self._prefetched[1]
            self._prefetched = None
        else:
            self._discard_prefetch()
            gathers = self._gather_layer(layer)

        # 第一轮记录真实执行顺序。后续最多预取下一层，此时它前两层
        # 的 forward 已经提交。顺序变化时，当前层改为按需收集。
        # max_norm 会原地更新权重，下一层如果共享该权重，应在更新后收集。
        mutates_weight = isinstance(layer, nn.Embedding) and layer.max_norm is not None
        if not mutates_weight and position < len(self._forward_order) - 1 and self._forward_order[position] is layer:
            next_layer = self._forward_order[position + 1]
            self._prefetched = (next_layer, self._gather_layer(next_layer))
        return gathers

    def _wrap_layer(self, layer: nn.Module, params: dict[str, nn.Parameter]) -> None:
        forward = layer.forward
        is_linear = isinstance(layer, (nn.Linear, Linear))

        @wraps(forward)
        def sharded_forward(*args, **kwargs):
            full_params = {name: gather.wait() for name, gather in self._prepare_layer(layer).items()}
            storages = {weight.untyped_storage().data_ptr(): self._shards[params[name]] for name, weight in full_params.items() if weight.numel()}

            def pack(tensor: Tensor) -> Tensor | _SavedWeight:
                shard = storages.get(tensor.untyped_storage().data_ptr())
                if shard is None:
                    return tensor.detach()
                # Linear 往往保存的是权重的转置。只保留视图信息，
                # 使得 forward 返回后可以释放完整权重。
                return _SavedWeight(shard, tensor.shape, tensor.stride(), int(tensor.storage_offset()))

            if is_linear and self.compute_dtype is not None:
                if args:
                    args = (args[0].to(self.compute_dtype), *args[1:])
                else:
                    name = "input" if isinstance(layer, nn.Linear) else "x"
                    if name in kwargs:
                        kwargs[name] = kwargs[name].to(self.compute_dtype)

            max_norm = layer.max_norm if isinstance(layer, nn.Embedding) else None
            try:
                if isinstance(layer, nn.Embedding) and max_norm is not None:
                    indices = args[0] if args else kwargs["input"]
                    self._renorm_embedding(layer, indices, full_params["weight"], max_norm)
                    layer.max_norm = None

                # 在本次调用内保留原层的 forward 实现，调用结束后恢复
                # 优化器一直持有的分片 Parameter。
                for name, weight in full_params.items():
                    layer._parameters[name] = _UnshardWeight.apply(params[name], weight, self)
                with saved_tensors_hooks(pack, self._unpack):
                    return forward(*args, **kwargs)
            finally:
                layer._parameters.update(params)
                if isinstance(layer, nn.Embedding) and max_norm is not None:
                    layer.max_norm = max_norm

        layer.forward = sharded_forward

    @torch.no_grad()
    def _renorm_embedding(self, layer: nn.Embedding, indices: Tensor, weight: Tensor, max_norm: float) -> None:
        # max_norm 修改所有输入 token 对应的行。取各 rank 的并集，让每个
        # rank 都能把变化写回自己的主权重分片。
        if self.world_size > 1:
            used = weight.new_zeros(weight.size(0), dtype=torch.uint8)
            used.index_fill_(0, indices.reshape(-1).long(), 1)
            dist.all_reduce(used, op=dist.ReduceOp.MAX)
            indices = used.nonzero().flatten()
        torch.embedding_renorm_(weight, indices.contiguous(), max_norm, layer.norm_type)
        parameter = self._layers[layer]["weight"]
        start = self._rank * parameter.numel()
        updated = weight.view(-1)[start : start + parameter.numel()]
        master = parameter[: updated.numel()]
        # 混合精度下只写回被 renorm 修改的元素，保留其它元素的主权重精度。
        master.copy_(torch.where(updated != master.to(weight.dtype), updated.to(master.dtype), master))

    def _unpack(self, saved: Tensor | _SavedWeight) -> Tensor:
        if isinstance(saved, Tensor):
            return saved
        full_weight = self._gather(saved.shard, self.compute_dtype).wait()
        return full_weight.as_strided(saved.shape, saved.stride, saved.offset)

    def forward(self, *inputs, **kwargs):
        self._current_order = []
        if self._forward_order:
            first = self._forward_order[0]
            self._prefetched = (first, self._gather_layer(first))
        try:
            result = self.module(*inputs, **kwargs)
        finally:
            self._discard_prefetch()
        self._forward_order = self._current_order
        return result

    def _reduce_shard_gradient(self, parameter: nn.Parameter, gradient: Tensor) -> None:
        if self._uses_nccl:
            device = gradient.device
            if device not in self._gradient_streams:
                self._gradient_streams[device] = torch.cuda.Stream(device=device)
            stream = self._gradient_streams[device]
            stream.wait_stream(torch.cuda.current_stream(device))
            # 在归约流等待 NCCL，避免反向计算流等待，也让完整梯度尽早可回收。
            gradient.record_stream(stream)
            with torch.cuda.stream(stream):
                local, work = self._reduce_scatter(parameter, gradient)
                if work is not None:
                    work.wait()
            self._pending_gradients.append((parameter, local, None))
        else:
            local, work = self._reduce_scatter(parameter, gradient)
            self._pending_gradients.append((parameter, local, work))

    def _reduce_scatter(self, parameter: nn.Parameter, gradient: Tensor) -> tuple[Tensor, dist.Work | None]:
        if gradient.is_sparse:
            gradient = gradient.to_dense()
        flat = gradient.to(dtype=parameter.dtype, memory_format=torch.contiguous_format).view(-1)
        if self.world_size == 1 or not parameter.numel():
            return flat, None
        padding = parameter.numel() * self.world_size - flat.numel()
        if padding:
            flat = nn.functional.pad(flat, (0, padding))
        local = torch.empty_like(parameter)
        return local, dist.reduce_scatter_tensor(local, flat, async_op=True)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """将通信结果同步到当前流，并写入 FP32 主权重的梯度。"""
        replicated = []
        if self.world_size > 1:
            for param in self._replicated:
                if param.grad is not None and param.grad.numel():
                    replicated.append((param.grad, dist.all_reduce(param.grad, async_op=True)))

        for device, stream in self._gradient_streams.items():
            torch.cuda.current_stream(device).wait_stream(stream)
        for param, grad, work in self._pending_gradients:
            if work is not None:
                work.wait()
            if self._uses_nccl:
                grad.record_stream(torch.cuda.current_stream(grad.device))
            grad.div_(self.world_size)
            if param.grad is None:
                param.grad = grad
            else:
                param.grad.add_(grad)
        self._pending_gradients.clear()

        for grad, work in replicated:
            if work is not None:
                work.wait()
            grad.div_(self.world_size)

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, Tensor]:
        """以主权重精度重建参数，供检查训练结果和导出使用。"""
        return {name: self._gather(self._shards[param], param.dtype).wait() if param in self._shards else param.detach() for name, param in self.module.named_parameters()}

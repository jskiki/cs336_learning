"""分布式数据并行的梯度同步实现。"""

from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def synchronize_gradients(self) -> None:
        """在 backward 后调用，各 rank 须对相同的参数产生梯度。"""
        for parameter in self.module.parameters():
            grad = parameter.grad
            if grad is not None:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(self.world_size)


class FlatDDP(NaiveDDP):
    """把同一设备、同一 dtype 的稠密梯度合并为一次 all-reduce。"""

    @torch.no_grad()
    def synchronize_gradients(self) -> None:
        """在 backward 后调用，各 rank 须对相同的参数产生梯度。"""
        grads = [grad for parameter in self.module.parameters() if (grad := parameter.grad) is not None]
        if not grads:
            return

        first = grads[0]
        if any(grad.device != first.device or grad.dtype != first.dtype or grad.layout != torch.strided for grad in grads):
            raise ValueError("梯度必须是同一设备、同一 dtype 的稠密张量") #与之相对的有稀疏coco梯度，存非零位置的梯度，坐标+值

        flat_grad = torch.cat([grad.reshape(-1) for grad in grads])
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad.div_(self.world_size)

        #按原来大小切开，view_as恢复形状
        chunks = flat_grad.split([grad.numel() for grad in grads])
        for grad, chunk in zip(grads, chunks, strict=True):
            grad.copy_(chunk.view_as(grad))


class OverlapDDP(NaiveDDP):
    """在反向传播期间异步同步梯度。各 rank 每步须使用相同的参数集合。"""

    def __init__(self, module: nn.Module, *, gradient_owner: Callable[[nn.Parameter], int] | None = None) -> None:
        """提供 gradient_owner 时，只将平均梯度交给负责更新该参数的 rank。"""
        super().__init__(module)
        self._rank = dist.get_rank()
        # 按固定顺序提交通信，避免不同 rank 的钩子触发顺序不同时归约错配。
        # 逆参数注册顺序通常更接近逐层反向传播的顺序。
        self._sync_order = [
            (param, gradient_owner(param) if gradient_owner is not None else None)
            for param in reversed(list(module.parameters()))
            if param.requires_grad
        ]
        #用于保存已经算完还没有提交通信的梯度
        self._ready_gradients: dict[nn.Parameter, torch.Tensor] = {}
        # 保存通信句柄，以及本 rank 需要取平均的梯度。
        self._pending_reductions: list[tuple[torch.Tensor | None, dist.Work]] = []
        #_sync_order 中下一个需要处理的位置。
        self._next_param = 0
        #给每个可训练的参数注册钩子（当反向传播更新某个参数时自动调用后面传入的函数）
        for param, _ in self._sync_order:
            param.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _on_grad_ready(self, param: nn.Parameter) -> None:
        grad = param.grad
        assert grad is not None
        self._ready_gradients[param] = grad
        self._launch_ready_gradients()

    def _launch_ready_gradients(self, *, backward_finished: bool = False) -> None:
        while self._next_param < len(self._sync_order):
            param, owner = self._sync_order[self._next_param]
            grad = self._ready_gradients.pop(param, None)
            if grad is None and not backward_finished:
                return

            self._next_param += 1
            if grad is not None:
                if owner is None:
                    work = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                else:
                    work = dist.reduce(grad, dst=owner, op=dist.ReduceOp.SUM, async_op=True)
                assert work is not None
                average_grad = grad if owner is None or owner == self._rank else None
                self._pending_reductions.append((average_grad, work))

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """在每次 backward 之后、optimizer.step 之前调用。"""
        # 反向已经结束，跳过所有 rank 都未使用的参数，提交排在它们之后的梯度。
        self._launch_ready_gradients(backward_finished=True)
        for grad, work in self._pending_reductions:
            work.wait()
            if grad is not None:
                grad.div_(self.world_size)

        self._pending_reductions.clear()
        self._next_param = 0

    synchronize_gradients = finish_gradient_synchronization

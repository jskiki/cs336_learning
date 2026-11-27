"""Overlapped data parallelism and sharded optimizer state."""

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn, no_grad
from torch.optim import Optimizer


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


class ShardedOptimizer(Optimizer):
    """外层管理完整参数组，内层优化器管理本 rank 的参数分片。

    各 rank 须以相同顺序注册参数组，并在 step 前准备好同步后的梯度。
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        #每个元素已分配的参数元素数
        self._elements_by_rank = [0] * self.world_size
        #记录参数张量和负责更新他的rank
        self._owners: dict[Tensor, int] = {}
        #收集当前rank的参数组，用来创建内层优化器
        local_groups: list[dict[str, Any]] = []
        self._initial_local_groups: list[dict[str, Any]] | None = local_groups

        # 父类构造时会调用 add_param_group。参数归属所需的状态应提前准备。
        super().__init__(params, defaults=kwargs.copy())

        # TODO：用本 rank 的参数组创建内层 optimizer_cls，并传递 kwargs。
        self._local_optimizer = optimizer_cls(local_groups, **kwargs)
        self._initial_local_groups = None
        self.defaults = self._local_optimizer.defaults
        self.state = self._local_optimizer.state
        #把封装的优化器的默认参数同步到外层，为了封装的外层也有完整的配置，不同步其实也能执行更新
        for group, local_group in zip(self.param_groups, self._local_optimizer.param_groups, strict=True):
            for key, value in local_group.items():
                if key not in ("params", "param_names"):
                    group.setdefault(key, value) #还有一次判断，缺少时才会填入

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """注册完整参数组，供外层 zero_grad 和学习率调度使用。"""
        group = param_group.copy()
        super().add_param_group(group)

        # TODO：为新增参数确定归属，记录各 rank 一致的同步顺序。
        # 在加组时完成分配，训练步复用这个结果。
        elements_by_rank = self._elements_by_rank.copy()
        owners: dict[Tensor, int] = {}
        for param in group["params"]:
            if param not in owners:
                owner = elements_by_rank.index(min(elements_by_rank))
                owners[param] = owner
                elements_by_rank[owner] += param.numel()

        # TODO：为本 rank 生成对应参数组，保留该组的超参数。
        local_indices = [index for index, param in enumerate(group["params"]) if owners[param] == self.rank]
        local_group = group | {"params": [group["params"][index] for index in local_indices]}
        if "param_names" in group:
            local_group["param_names"] = [group["param_names"][index] for index in local_indices]

        if self._initial_local_groups is not None:
            self._initial_local_groups.append(local_group)
        else:
            try:
                self._local_optimizer.add_param_group(local_group)
            except Exception:
                self.param_groups.pop()
                raise
        self._owners.update(owners)
        self._elements_by_rank = elements_by_rank

    def step(self, closure: Callable[[], Any] | None = None, **kwargs: Any) -> Any:
        """更新本地分片，随后同步更新后的模型参数。"""
        # TODO：将当前参数组设置、closure 和 kwargs 传给内层优化器。
        for group, local_group in zip(self.param_groups, self._local_optimizer.param_groups, strict=True):
            for key, value in group.items():
                if key not in ("params", "param_names"):
                    local_group[key] = value
        result = self._local_optimizer.step(closure, **kwargs)

        # TODO：沿已记录的顺序广播更新后的权重，并返回内层 step 的结果。
        if self.world_size > 1:
            with no_grad():
                pending = []
                for param, owner in self._owners.items():
                    work = dist.broadcast(param, src=owner, async_op=True)
                    assert work is not None
                    pending.append(work)
                for work in pending:
                    work.wait()
        return result

    def parameter_owner(self, param: Tensor) -> int:
        """返回负责更新该参数的 rank，供 OverlapDDP 的 gradient_owner 回调使用。"""
        return self._owners[param]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """恢复同一 rank 上保存的优化器状态。"""
        super().load_state_dict(state_dict | {"state": {}})
        local_groups = []
        local_ids = set()
        parameter_ids = set()
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            indices = [i for i, param in enumerate(group["params"]) if self._owners[param] == self.rank]
            saved_ids = [saved_group["params"][i] for i in indices]
            local_group = saved_group | {"params": saved_ids}
            if "param_names" in saved_group:
                local_group["param_names"] = [saved_group["param_names"][i] for i in indices]
            local_groups.append(local_group)
            local_ids.update(saved_ids)
            parameter_ids.update(saved_group["params"])
        local_state = {key: value for key, value in state_dict["state"].items() if key in local_ids or key not in parameter_ids}
        self._local_optimizer.load_state_dict({"state": local_state, "param_groups": local_groups})
        self.state = self._local_optimizer.state

"""按 rank 分摊优化器状态，并在更新后同步模型参数。"""

from collections.abc import Callable, Iterable
from typing import Any

import torch.distributed as dist
from torch import Tensor, no_grad
from torch.optim import Optimizer


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

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """恢复同一 rank 上保存的优化器状态。"""
        super().load_state_dict(state_dict)
        self._local_optimizer.state = self.state

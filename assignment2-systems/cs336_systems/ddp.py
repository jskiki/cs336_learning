"""在反向结束后逐参数同步梯度的朴素 DDP。"""

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

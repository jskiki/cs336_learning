"""作业 4.3：AdamW 优化器框架，核心计算由你完成。

测试入口：tests/adapters.py 中的 get_adamw_cls。
完成后，在 assignment1-basics 目录运行：uv run pytest -k test_adamw。
"""

from collections.abc import Callable

import torch
from torch import Tensor
from math import sqrt


class AdamW(torch.optim.Optimizer):
    """维护每个参数的一阶、二阶矩估计，并执行解耦的权重衰减。"""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """注册参数和默认超参数。

        params: 参数集合，或包含 params 和超参数的参数组集合。
        lr: 题目中的初始学习率 alpha。
        betas: 题目中的 (beta_1, beta_2)。
        eps: 题目中的数值稳定项 epsilon。
        weight_decay: 题目中的权重衰减率 lambda。

        self.param_groups 由父类管理，每个参数组可覆盖默认超参数。
        self.state[p] 保存参数 p 的状态，建议使用 t、m、v 作为键。
        m、v 应与 p 的形状、设备和数据类型一致。
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    #装饰器，执行这个函数时，不记录其中运算的自动求导计算图
    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """执行一次更新；梯度应已由外部的 loss.backward() 计算。

        参数更新不记录计算图；closure 如需重新计算损失，则启用梯度。
        不在此处清除梯度，不替换参数对象。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        #group是一个字典，里面有参数和优化参数需要的超参数
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            # TODO：读取当前参数组的 lr、betas、eps 和 weight_decay。
            for p in group["params"]:
                if p.grad is None:
                    continue

                # TODO 1：首次使用参数时，在 self.state[p] 中初始化 t、m、v。
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                # TODO 2：维护当前参数的更新次数；公式中的首次更新 t 为 1。
                t = state["t"]
                m = state["m"]
                v = state["v"]
                t = t + 1
                # TODO 3：按题目计算偏差修正后的学习率。
                #这里已经计算了偏差修正，后面直接用m和v就行
                lr_t = lr*sqrt(1-beta2**t)/(1-beta1**t)
                # TODO 4：按题目顺序应用解耦的权重衰减。
                grad = p.grad
                p.data -= lr*weight_decay*p.data
                # TODO 5：根据 p.grad 更新一阶和二阶矩估计。
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2
                # TODO 6：根据矩估计原地更新参数，并保存状态。
                p.data -= lr_t*m/(v.sqrt()+eps)
                state["t"] = t
                state["m"] = m
                state["v"] = v
                
        return loss

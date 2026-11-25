"""AdamW、梯度裁剪和余弦学习率调度。"""

from collections.abc import Callable, Iterable
from math import cos, pi, sqrt
from typing import Any

import torch
from torch import no_grad, stack
from torch.linalg import vector_norm
from torch.nn import Parameter


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

    # 装饰器，执行这个函数时，不记录其中运算的自动求导计算图
    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        """执行一次更新；梯度应已由外部的 loss.backward() 计算。

        参数更新不记录计算图；closure 如需重新计算损失，则启用梯度。
        不在此处清除梯度，不替换参数对象。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # group是一个字典，里面有参数和优化参数需要的超参数
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
                # 这里已经计算了偏差修正，后面直接用m和v就行
                lr_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
                # TODO 4：按题目顺序原地应用解耦的权重衰减，避免创建额外临时张量。
                grad = p.grad
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                # TODO 5：根据 p.grad 原地更新一阶和二阶矩估计。
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # TODO 6：根据矩估计原地更新参数，并保存更新计数。
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr_t)
                state["t"] = t

        return loss


@no_grad()
def gradient_clipping(parameters: Iterable[Parameter], max_l2_norm: float) -> None:
    """按全局 L2 范数原地裁剪参数梯度，不修改参数值。

    参数：
        parameters: 参数列表、元组或 model.parameters() 等可迭代对象。
            注意，传入的对象可能是只能遍历一次的迭代器。
        max_l2_norm: 允许的最大全局梯度 L2 范数，对应题目中的 M。

    返回值：
        None；直接修改各参数的 .grad。

    完成标准：
        在 loss.backward() 之后、optimizer.step() 之前调用。
        忽略 .grad 为 None 的参数；没有梯度时不做任何修改。
        计算全部有效梯度共同的范数，而不是分别裁剪各个参数。
        需要裁剪时，对所有有效梯度使用相同的缩放系数。
        按题目使用 epsilon = 1e-6，且不放大原本较小的梯度。
        原地修改梯度，保持梯度的形状、设备和数据类型不变。
    """
    # TODO 1：收集有效梯度，并处理没有有效梯度的情况。
    gradients = [p.grad for p in parameters if p.grad is not None]
    # TODO 2：计算所有有效梯度共同的 L2 范数。
    # 分别计算每个梯度张量的L2范数，stack把他们堆叠成一个张量，最后外层再计算一次
    total_norm = vector_norm(stack([vector_norm(grad) for grad in gradients]))
    # TODO 3：按题目确定是否裁剪及缩放系数。
    # clamp 是 PyTorch 张量 torch.Tensor 的方法，用于把元素限制在指定范围内,x.clamp(min=0.0, max=1.0),这里已经知道肯定为正数，则不设min
    scale = (max_l2_norm / (total_norm + 1e-6)).clamp(max=1.0)
    # TODO 4：原地缩放梯度，不修改参数本身。
    for grad in gradients:
        grad.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """返回指定迭代步数的基础学习率，不直接修改优化器。

    参数：
        it: 当前迭代步数，对应题目中的 t，从 0 开始。
        max_learning_rate: 最大学习率，对应 alpha_max。
        min_learning_rate: 最终学习率，对应 alpha_min。
        warmup_iters: 预热迭代次数，对应 T_w。
        cosine_cycle_iters: 余弦退火结束时的步数，对应 T_c。
            这是从训练开始计数的结束步数，不是退火阶段的持续步数。

    返回值：
        当前步的学习率，类型为 Python float。

    输入约定：
        it >= 0，0 <= warmup_iters < cosine_cycle_iters。
        0 <= min_learning_rate <= max_learning_rate。

    完成标准：
        预热阶段按题目要求线性增加学习率。
        预热结束后，按余弦曲线衰减到最小学习率。
        退火结束后保持最小学习率。
        注意两个阶段交界处，以及 warmup_iters 为 0 时的行为。
    """
    # TODO 1：处理预热阶段。
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # TODO 2：处理余弦退火阶段。
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + cos(pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (
            max_learning_rate - min_learning_rate
        )
    # TODO 3：处理退火结束后的阶段。
    return min_learning_rate

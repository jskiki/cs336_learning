"""作业 4.5：梯度裁剪框架，核心计算由你完成。

测试入口：tests/adapters.py 中的 run_gradient_clipping。
完成后，在 assignment1-basics 目录运行：uv run pytest -k test_gradient_clipping。
"""

from collections.abc import Iterable

from torch import no_grad, stack
from torch.linalg import vector_norm
from torch.nn import Parameter


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
    gradients = [p.grad  for p in parameters if p.grad is not None]
    # TODO 2：计算所有有效梯度共同的 L2 范数。
    #分别计算每个梯度张量的L2范数，stack把他们堆叠成一个张量，最后外层再计算一次
    total_norm = vector_norm(
        stack([vector_norm(grad) for grad in gradients])
    )
    # TODO 3：按题目确定是否裁剪及缩放系数。
    #clamp 是 PyTorch 张量 torch.Tensor 的方法，用于把元素限制在指定范围内,x.clamp(min=0.0, max=1.0),这里已经知道肯定为正数，则不设min
    scale = (max_l2_norm / (total_norm + 1e-6)).clamp(max=1.0) 
    # TODO 4：原地缩放梯度，不修改参数本身。
    for grad in gradients:
        grad.mul_(scale)

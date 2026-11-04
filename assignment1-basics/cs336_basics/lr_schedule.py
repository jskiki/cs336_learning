"""作业 4.4：带预热的余弦学习率调度框架，核心计算由你完成。

测试入口：tests/adapters.py 中的 run_get_lr_cosine_schedule。
完成后，在 assignment1-basics 目录运行：
    uv run pytest -k test_get_lr_cosine_schedule
"""
from math import pi,cos

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
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    # TODO 2：处理余弦退火阶段。
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5*(1+cos(pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    # TODO 3：处理退火结束后的阶段。
    return min_learning_rate

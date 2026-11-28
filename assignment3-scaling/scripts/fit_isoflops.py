"""CS336 Assignment 3 Section 2.1 (chinchilla_isoflops): 基于 IsoFLOPs 曲线拟合缩放定律。

核心步骤：
1. 读取 data/isoflops_curves.json 中的训练运行记录，并按 compute_budget（C）分组。
2. 对每个固定计算预算 C_i，选取 final_loss 最低的运行作为最优点，得到：
   - 最优模型参数量 N_opt(C_i)
   - 最优训练 token 数 D_opt(C_i) = C_i / (6 * N_opt(C_i))
3. 在 log10 空间中对幂律 N_opt(C) = k_N * C^a 与 D_opt(C) = k_D * C^b 做线性拟合
   （使用 scipy.optimize.curve_fit），避免大算力点垄断残差并严格保持 C = 6ND。
4. 外推预测 C = 10^23 与 C = 10^24 FLOPs 下的最优模型规模与数据集规模，并生成可视化图表。
5. 额外打印“每条 IsoFLOPs 曲线二次拟合找谷底（Chinchilla 原论文法）”与“原始空间非线性拟合”的对照结果。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from scipy.optimize import curve_fit

# 使用非交互式后端并配置中文字体，确保在无图形界面的终端/服务器环境中直接导出中文图表
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    *matplotlib.rcParams["font.sans-serif"],
]

# 题目要求外推评估的目标计算预算（FLOPs）
TARGET_BUDGETS: tuple[float, ...] = (1e23, 1e24)


@dataclass(frozen=True)
class PowerLawFit:
    """幂律拟合结果：y(C) = k * C^exponent。"""

    k: float
    exponent: float
    r2_log: float
    exponent_se: float

    def predict(self, compute_budget: float | np.ndarray) -> float | np.ndarray:
        """预测给定计算预算 C 下的最优规模（支持标量或 NumPy 数组）。"""
        return self.k * np.power(compute_budget, self.exponent)


def format_count(value: float, unit_suffix: str = "") -> str:
    """将大数格式化为易读的科学量级字符串（如 70.1B、808.6B、1.2T）。"""
    for scale, prefix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{prefix}{unit_suffix}"
    return f"{value:.2f}{unit_suffix}"


def format_flops(compute_budget: float) -> str:
    """将 FLOPs 格式化为简洁的幂次表示（如 10^23）。"""
    exp = int(round(np.log10(compute_budget)))
    mantissa = compute_budget / (10.0**exp)
    if np.isclose(mantissa, 1.0):
        return f"10^{exp}"
    return f"{mantissa:.1f}e{exp}"


def load_grouped_runs(json_path: Path) -> dict[float, list[dict[str, float]]]:
    """读取 JSON 训练日志并按 compute_budget 升序分组，每组内按 parameters 升序排列。"""
    runs = json.loads(json_path.read_text(encoding="utf-8"))
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        grouped[float(run["compute_budget"])].append(
            {
                "parameters": float(run["parameters"]),
                "compute_budget": float(run["compute_budget"]),
                "final_loss": float(run["final_loss"]),
            }
        )

    # 保证每个预算内部按模型参数量递增排序，方便后续画 U 形曲线和定位最优点下标
    return {
        budget: sorted(group, key=lambda item: item["parameters"])
        for budget, group in sorted(grouped.items())
    }


def extract_argmin_optima(
    grouped_runs: dict[float, list[dict[str, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """按题目推荐方法：对每个计算预算直接取 final_loss 最低的运行作为最优点。

    返回：
        budgets: 计算预算数组 C_i
        n_opts: 最优模型参数量 N_opt(C_i)
        d_opts: 最优训练 token 数 D_opt(C_i) = C_i / (6 * N_opt(C_i))
        min_losses: 各预算对应的最低 final_loss
        best_indices: 最优点在组内按参数量排序后的下标（用于 sanity check，确认不在边界）
    """
    budgets: list[float] = []
    n_opts: list[float] = []
    d_opts: list[float] = []
    min_losses: list[float] = []
    best_indices: list[int] = []

    for budget, group in grouped_runs.items():
        # 直接寻找该 IsoFLOPs 组内 final_loss 最小的训练记录
        best_idx, best_run = min(enumerate(group), key=lambda pair: pair[1]["final_loss"])
        n_opt = best_run["parameters"]
        # 根据 Transformer 训练近似算力公式 C = 6 * N * D 反推出训练数据量 D
        d_opt = budget / (6.0 * n_opt)

        budgets.append(budget)
        n_opts.append(n_opt)
        d_opts.append(d_opt)
        min_losses.append(best_run["final_loss"])
        best_indices.append(best_idx)

    return (
        np.asarray(budgets, dtype=np.float64),
        np.asarray(n_opts, dtype=np.float64),
        np.asarray(d_opts, dtype=np.float64),
        np.asarray(min_losses, dtype=np.float64),
        best_indices,
    )


def extract_quadratic_optima(
    grouped_runs: dict[float, list[dict[str, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对照方法（Chinchilla 原论文 Approach 2）：对每条 IsoFLOPs 曲线拟合二次抛物线求解析谷底。

    在 (log10(N), final_loss) 上拟合 L = p2 * x^2 + p1 * x + p0，
    当 p2 > 0 时抛物线开口向上，最小值点位于 x* = -p1 / (2 * p2)，即 N_opt = 10^(x*)。
    """
    budgets: list[float] = []
    n_opts: list[float] = []
    d_opts: list[float] = []

    for budget, group in grouped_runs.items():
        log_n = np.log10([run["parameters"] for run in group])
        losses = np.asarray([run["final_loss"] for run in group], dtype=np.float64)

        p2, p1, _ = np.polyfit(log_n, losses, deg=2)
        if p2 <= 0:
            raise ValueError(f"计算预算 {budget:.1e} 的二次拟合开口未向上（p2={p2:.4e}）")

        log_n_opt = -p1 / (2.0 * p2)
        n_opt = 10.0**log_n_opt
        d_opt = budget / (6.0 * n_opt)

        budgets.append(budget)
        n_opts.append(n_opt)
        d_opts.append(d_opt)

    return (
        np.asarray(budgets, dtype=np.float64),
        np.asarray(n_opts, dtype=np.float64),
        np.asarray(d_opts, dtype=np.float64),
    )


def fit_power_law_log_space(budgets: np.ndarray, targets: np.ndarray) -> PowerLawFit:
    """在对数空间使用 scipy.optimize.curve_fit 拟合幂律 y = k * C^exponent。

    为什么先取 log10 再拟合：
    1. 数据跨越多个数量级（C: 10^18~10^21），对数空间最小二乘等价于最小化相对误差，
       避免最大算力点主导整个拟合目标；
    2. 避免直接拟合 C^a 时由于数值量级过大导致的病态优化或收敛失败；
    3. 因为 log10(D) = log10(C) - log10(6) - log10(N) 是线性关系，
       对 N 和 D 分别做对数空间最小二乘会严格满足 a + b = 1 与 6 * k_N * k_D = 1。
    """
    log_c = np.log10(budgets)
    log_y = np.log10(targets)

    def linear_model(x: np.ndarray, log_k: float, exponent: float) -> np.ndarray:
        return log_k + exponent * x

    # curve_fit 返回最优参数 popt 和协方差矩阵 pcov
    popt, pcov = curve_fit(linear_model, log_c, log_y)
    log_k, exponent = float(popt[0]), float(popt[1])
    exponent_se = float(np.sqrt(pcov[1, 1]))

    # 计算对数空间决定系数 R^2，衡量数据点在 log-log 直线上的贴合度
    pred_log_y = linear_model(log_c, log_k, exponent)
    ss_res = float(np.sum((log_y - pred_log_y) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2_log = 1.0 - ss_res / ss_tot

    return PowerLawFit(
        k=10.0**log_k,
        exponent=exponent,
        r2_log=r2_log,
        exponent_se=exponent_se,
    )


def fit_power_law_raw_space(
    budgets: np.ndarray, targets: np.ndarray, init_fit: PowerLawFit
) -> PowerLawFit:
    """对照方法：直接在原始线性空间调用 curve_fit 拟合 y = k * C^exponent。

    为了避免数值量级过大导致优化失败，这里把 C 缩放为 C / C_ref，并用对数拟合结果作初值。
    """
    c_ref = float(np.median(budgets))
    c_scaled = budgets / c_ref
    y_ref = float(np.median(targets))
    y_scaled = targets / y_ref

    def scaled_power_law(x: np.ndarray, coef: float, exponent: float) -> np.ndarray:
        return coef * np.power(x, exponent)

    init_coef = float(init_fit.predict(c_ref) / y_ref)
    popt, pcov = curve_fit(
        scaled_power_law,
        c_scaled,
        y_scaled,
        p0=(init_coef, init_fit.exponent),
    )
    coef, exponent = float(popt[0]), float(popt[1])
    # 还原回原始标度：y = (coef * y_ref / c_ref^exponent) * C^exponent
    k = coef * y_ref / (c_ref**exponent)

    log_y = np.log10(targets)
    pred_log_y = np.log10(k * np.power(budgets, exponent))
    r2_log = 1.0 - float(np.sum((log_y - pred_log_y) ** 2)) / float(
        np.sum((log_y - np.mean(log_y)) ** 2)
    )
    return PowerLawFit(
        k=k,
        exponent=exponent,
        r2_log=r2_log,
        exponent_se=float(np.sqrt(pcov[1, 1])),
    )


def plot_scaling_law(
    budgets: np.ndarray,
    observed: np.ndarray,
    fit: PowerLawFit,
    *,
    symbol: str,
    ylabel: str,
    title: str,
    unit_label: str,
    save_path: Path,
    curve_color: str,
) -> None:
    """绘制单张双对数坐标缩放定律图，包含实测拟合点、拟合曲线和外推至 >10^24 FLOPs 的预测点。"""
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=220)

    # 区分实测拟合区间（实线）和外推区间（虚线），让外推范围一目了然
    c_fit_grid = np.geomspace(budgets.min() * 0.8, budgets.max(), 200)
    c_extrap_grid = np.geomspace(budgets.max(), 2.0e24, 200)

    eq_label = (
        rf"幂律拟合 ${symbol}_{{\mathrm{{opt}}}} = "
        rf"{fit.k:.3g} \cdot C^{{{fit.exponent:.4f}}}$ "
        rf"（$R^2_{{\log}} = {fit.r2_log:.4f}$）"
    )

    ax.plot(
        c_fit_grid,
        fit.predict(c_fit_grid),
        color=curve_color,
        linewidth=2.2,
        label=eq_label,
        zorder=2,
    )
    ax.plot(
        c_extrap_grid,
        fit.predict(c_extrap_grid),
        color=curve_color,
        linewidth=2.2,
        linestyle="--",
        label="外推区间",
        zorder=2,
    )

    # 实测最优点
    ax.scatter(
        budgets,
        observed,
        color="#1f2937",
        s=52,
        zorder=4,
        label=rf"实测最优点 $\langle C_i, {symbol}_{{\mathrm{{opt}}}}(C_i) \rangle$",
    )

    # 目标预算外推点与标注
    target_c = np.asarray(TARGET_BUDGETS, dtype=np.float64)
    target_y = np.asarray(fit.predict(target_c), dtype=np.float64)
    ax.scatter(
        target_c,
        target_y,
        color="#dc2626",
        marker=MarkerStyle("*"),
        s=200,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="预测最优点（$10^{23}$、$10^{24}$）",
    )

    # 将 10^23 标注放在曲线左上方留白区，10^24 标注放在 10^24 预测点正下方偏左留白区，避免遮挡星标
    offsets = [(-105, 28), (-55, -78)]
    for c_val, y_val, offset in zip(target_c, target_y, offsets, strict=True):
        exp = int(round(np.log10(c_val)))
        text = f"$C = 10^{{{exp}}}$\n{float(y_val) / 1e8:.1f} 亿{unit_label}"
        ax.annotate(
            text,
            xy=(c_val, y_val),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold",
            color="#991b1b",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#fef2f2",
                edgecolor="#fca5a5",
                linewidth=0.9,
            ),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.1),
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(budgets.min() * 0.5, 3.0e24)
    ax.set_ylim(observed.min() * 0.5, float(target_y.max()) * 2.8)
    ax.set_xlabel("计算预算 $C$（次浮点运算）", fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.set_title(title, fontsize=13, fontweight="semibold", pad=12)

    ax.grid(True, which="major", linestyle="-", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=220)
    plt.close(fig)


# Matplotlib 官方 viridis 调色板的 9 个均匀采样控制点，避免 Pylance 旧版内置 stub 对 cm/colormaps 报错
_VIRIDIS_RGB: tuple[tuple[float, float, float], ...] = (
    (0.267004, 0.004874, 0.329415),
    (0.278826, 0.175490, 0.483397),
    (0.229739, 0.322361, 0.545706),
    (0.172719, 0.448791, 0.557885),
    (0.127568, 0.566949, 0.550556),
    (0.157851, 0.683765, 0.501686),
    (0.369214, 0.788888, 0.382914),
    (0.678489, 0.863742, 0.189503),
    (0.993248, 0.906157, 0.143936),
)


def _viridis_hex(t: float) -> str:
    """按归一化位置 t ∈ [0, 1] 线性插值返回 viridis 十六进制颜色字符串。"""
    t_clamped = float(np.clip(t, 0.0, 1.0))
    stops = np.linspace(0.0, 1.0, len(_VIRIDIS_RGB))
    rgb_table = np.asarray(_VIRIDIS_RGB, dtype=np.float64)
    r = int(round(float(np.interp(t_clamped, stops, rgb_table[:, 0])) * 255))
    g = int(round(float(np.interp(t_clamped, stops, rgb_table[:, 1])) * 255))
    b = int(round(float(np.interp(t_clamped, stops, rgb_table[:, 2])) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_isoflops_profiles(
    grouped_runs: dict[float, list[dict[str, float]]],
    save_path: Path,
) -> None:
    """绘制 9 条原始 IsoFLOPs U 形曲线，并圈出每条曲线选中的最小 loss 点（辅助诊断图）。"""
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=220)
    budgets = list(grouped_runs.keys())

    for idx, budget in enumerate(budgets):
        group = grouped_runs[budget]
        params = np.asarray([run["parameters"] for run in group], dtype=np.float64)
        losses = np.asarray([run["final_loss"] for run in group], dtype=np.float64)
        best_idx = int(np.argmin(losses))
        color = _viridis_hex(idx / max(1, len(budgets) - 1))

        exp = int(np.floor(np.log10(budget)))
        mantissa = budget / (10.0**exp)
        budget_label = (
            rf"$C = 10^{{{exp}}}$"
            if np.isclose(mantissa, 1.0)
            else rf"$C = {mantissa:.0f} \times 10^{{{exp}}}$"
        )
        ax.plot(
            params,
            losses,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=color,
            label=budget_label,
        )
        # 用红边星标突出每条 IsoFLOPs 曲线的最低 loss 点
        ax.scatter(
            params[best_idx],
            losses[best_idx],
            color=color,
            edgecolor="#dc2626",
            linewidth=1.6,
            s=95,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("模型参数量 $N$（个）", fontsize=11.5)
    ax.set_ylabel("最终训练损失 $L$", fontsize=11.5)
    ax.set_title(
        "等计算量曲线与最小损失点（红圈）",
        fontsize=12.5,
        fontweight="semibold",
        pad=12,
    )
    ax.grid(True, which="major", linestyle="-", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="计算预算（次浮点运算）", ncol=2, fontsize=8.5, title_fontsize=9, loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=220)
    plt.close(fig)


def print_summary(
    grouped_runs: dict[float, list[dict[str, float]]],
    budgets: np.ndarray,
    n_opts: np.ndarray,
    d_opts: np.ndarray,
    min_losses: np.ndarray,
    best_indices: list[int],
    n_fit: PowerLawFit,
    d_fit: PowerLawFit,
    n_fit_quad: PowerLawFit,
    d_fit_quad: PowerLawFit,
    n_fit_raw: PowerLawFit,
    d_fit_raw: PowerLawFit,
) -> None:
    """打印各预算最优点表、拟合参数、自洽性检查与不同拟合方式的横向对比。"""
    print("=" * 86)
    print("1. 各计算预算下选取的最优点（直接取最小 final_loss）")
    print("=" * 86)
    header = (
        f"{'Compute C (FLOPs)':>18} | {'Idx':>5} | {'N_opt':>14} | "
        f"{'D_opt (tokens)':>15} | {'D/N':>6} | {'Min Loss':>8}"
    )
    print(header)
    print("-" * len(header))
    for c, idx, n_val, d_val, loss_val in zip(
        budgets, best_indices, n_opts, d_opts, min_losses, strict=True
    ):
        group_size = len(grouped_runs[float(c)])
        print(
            f"{c:18.2e} | {idx + 1:>2}/{group_size:<2} | "
            f"{n_val:14.4e} | {d_val:15.4e} | {d_val / n_val:6.2f} | {loss_val:8.4f}"
        )

    print("\n" + "=" * 86)
    print("2. 主方法（题目推荐 Argmin + log10 空间 curve_fit）拟合与外推结果")
    print("=" * 86)
    print(
        f"模型规模缩放律: N_opt(C) = {n_fit.k:.6e} * C^{n_fit.exponent:.6f} "
        f"(±{n_fit.exponent_se:.4f}, log-R^2 = {n_fit.r2_log:.6f})"
    )
    print(
        f"数据规模缩放律: D_opt(C) = {d_fit.k:.6e} * C^{d_fit.exponent:.6f} "
        f"(±{d_fit.exponent_se:.4f}, log-R^2 = {d_fit.r2_log:.6f})"
    )
    print(
        f"自洽性检查:     a + b = {n_fit.exponent + d_fit.exponent:.10f}, "
        f"6 * k_N * k_D = {6.0 * n_fit.k * d_fit.k:.10f}"
    )

    for target in TARGET_BUDGETS:
        pred_n = float(n_fit.predict(target))
        pred_d = float(d_fit.predict(target))
        ratio = 6.0 * pred_n * pred_d / target
        print(
            f"  C = {format_flops(target):>5} FLOPs -> "
            f"N_opt = {pred_n:.4e} ({format_count(pred_n)}), "
            f"D_opt = {pred_d:.4e} ({format_count(pred_d)}), "
            f"D/N = {pred_d / pred_n:.2f}, 6ND/C = {ratio:.6f}"
        )

    print("\n" + "=" * 86)
    print("3. 扩展对照：Chinchilla 二次拟合求谷底 vs 原始线性空间 curve_fit")
    print("=" * 86)
    for label, nf, df in (
        ("Chinchilla 二次拟合谷底 + log10 拟合", n_fit_quad, d_fit_quad),
        ("Argmin + 原始线性空间直接 curve_fit ", n_fit_raw, d_fit_raw),
    ):
        pred_n_23, pred_n_24 = float(nf.predict(1e23)), float(nf.predict(1e24))
        pred_d_23, pred_d_24 = float(df.predict(1e23)), float(df.predict(1e24))
        print(f"[{label}]")
        print(
            f"  a = {nf.exponent:.6f} (R^2={nf.r2_log:.6f}), "
            f"b = {df.exponent:.6f} (R^2={df.r2_log:.6f}), "
            f"a + b = {nf.exponent + df.exponent:.6f}"
        )
        print(
            f"  10^23 FLOPs: N = {format_count(pred_n_23)}, D = {format_count(pred_d_23)}, "
            f"6ND/C = {6 * pred_n_23 * pred_d_23 / 1e23:.4f}"
        )
        print(
            f"  10^24 FLOPs: N = {format_count(pred_n_24)}, D = {format_count(pred_d_24)}, "
            f"6ND/C = {6 * pred_n_24 * pred_d_24 / 1e24:.4f}"
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="复现 Chinchilla IsoFLOPs 缩放定律拟合与外推（CS336 A3 Section 2.1）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "isoflops_curves.json",
        help="IsoFLOPs 训练运行数据 JSON 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "Note" / "figures",
        help="输出图表的保存目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped_runs = load_grouped_runs(args.input)

    # 1. 主方法：题目推荐的直接取每组最小 loss 点 + 对数空间幂律拟合
    budgets, n_opts, d_opts, min_losses, best_indices = extract_argmin_optima(grouped_runs)
    n_fit = fit_power_law_log_space(budgets, n_opts)
    d_fit = fit_power_law_log_space(budgets, d_opts)

    # 2. 对照实验：Chinchilla 二次抛物线求谷底 + 对数拟合；以及原始空间直接拟合
    budgets_q, n_opts_q, d_opts_q = extract_quadratic_optima(grouped_runs)
    n_fit_quad = fit_power_law_log_space(budgets_q, n_opts_q)
    d_fit_quad = fit_power_law_log_space(budgets_q, d_opts_q)
    n_fit_raw = fit_power_law_raw_space(budgets, n_opts, init_fit=n_fit)
    d_fit_raw = fit_power_law_raw_space(budgets, d_opts, init_fit=d_fit)

    # 3. 输出完整数值分析
    print_summary(
        grouped_runs=grouped_runs,
        budgets=budgets,
        n_opts=n_opts,
        d_opts=d_opts,
        min_losses=min_losses,
        best_indices=best_indices,
        n_fit=n_fit,
        d_fit=d_fit,
        n_fit_quad=n_fit_quad,
        d_fit_quad=d_fit_quad,
        n_fit_raw=n_fit_raw,
        d_fit_raw=d_fit_raw,
    )

    # 4. 生成题目要求的 (a)、(b) 两张外推图，以及辅助诊断用的 IsoFLOPs U 形曲线图
    model_plot = args.output_dir / "chinchilla_isoflops_model_size.png"
    dataset_plot = args.output_dir / "chinchilla_isoflops_dataset_size.png"
    profiles_plot = args.output_dir / "chinchilla_isoflops_profiles.png"

    plot_scaling_law(
        budgets,
        n_opts,
        n_fit,
        symbol="N",
        ylabel=r"算力最优模型参数量 $N_{\mathrm{opt}}$（个）",
        title="算力最优模型规模的缩放定律",
        unit_label="参数",
        save_path=model_plot,
        curve_color="#2563eb",
    )
    plot_scaling_law(
        budgets,
        d_opts,
        d_fit,
        symbol="D",
        ylabel=r"算力最优训练词元数 $D_{\mathrm{opt}}$（个）",
        title="算力最优数据集规模的缩放定律",
        unit_label="词元",
        save_path=dataset_plot,
        curve_color="#059669",
    )
    plot_isoflops_profiles(grouped_runs, save_path=profiles_plot)

    print("\n图表已导出：")
    print(f"  - {model_plot}")
    print(f"  - {dataset_plot}")
    print(f"  - {profiles_plot}")


if __name__ == "__main__":
    main()

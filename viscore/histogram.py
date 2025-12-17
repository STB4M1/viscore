"""
VisCore Histogram Utility
-------------------------
任意の 1 次元データに対して、VisCore 統一スタイルのヒストグラムを描画する関数。

特徴：
 - ビン幅自動計算（Sturges / FD / Scott / Robust Scott / sqrt / Shimazaki）
 - KDE オプション（ON/OFF, 色・線幅指定可能）
 - 外観を引数で完全制御
 - SVG/PNG 保存対応
 - VisCore フォント & スタイル自動適用
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.stats import gaussian_kde

from viscore.styles.tex_fonts import setup_fonts


# ===============================================
#  ビン幅計算（内部ユーティリティ）
# ===============================================
def _calc_bins(data, method="fd"):

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    x_min, x_max = data.min(), data.max()
    method = method.lower()

    if method == "sturges":
        k = int(round(1 + np.log2(len(data))))
        return np.linspace(x_min, x_max, k + 1)

    if method == "fd":
        IQR = np.percentile(data, 75) - np.percentile(data, 25)
        h = 2 * IQR * len(data) ** (-1/3)
        k = int(np.ceil((x_max - x_min) / h)) if h > 0 else int(np.sqrt(len(data)))
        return np.linspace(x_min, x_max, k + 1)

    if method == "sqrt":
        k = int(np.sqrt(len(data)))
        return np.linspace(x_min, x_max, k + 1)

    if method == "scott":
        s = np.std(data, ddof=1)
        h = 3.49 * s * len(data) ** (-1/3)
        k = int(np.ceil((x_max - x_min) / h)) if h > 0 else int(np.sqrt(len(data)))
        return np.linspace(x_min, x_max, k + 1)

    if method in ("scott_robust", "robust_scott"):
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        sigma = 1.4826 * mad
        h = 3.49 * sigma * len(data) ** (-1/3)
        k = int(np.ceil((x_max - x_min) / h)) if h > 0 else int(np.sqrt(len(data)))
        return np.linspace(x_min, x_max, k + 1)

    if method == "shimazaki":
        Kmin, Kmax = 4, 150
        ks = np.arange(Kmin, Kmax + 1)
        costs = []

        for k in ks:
            edges = np.linspace(x_min, x_max, k + 1)
            counts, _ = np.histogram(data, bins=edges)
            v = np.var(counts)
            m = np.mean(counts)
            delta = (x_max - x_min) / k
            costs.append((2*m - v) / (delta**2))

        k_opt = ks[np.argmin(costs)]
        return np.linspace(x_min, x_max, k_opt + 1)

    raise ValueError(f"Unknown binning method: {method}")


# ===============================================
#  ヒストグラム描画（KDE 対応）
# ===============================================
def plot_histogram(
    data,
    bins="fd",
    range=None,

    # === Appearance ===
    figsize=(8, 5),
    dpi=200,
    facecolor="royalblue",
    edgecolor="black",
    alpha=0.8,

    # === KDE ===
    kde=False,
    kde_color="black",
    kde_linewidth=2.2,
    kde_linestyle="-",

    # === Labels / Title ===
    title=None,
    title_fontsize=28,
    xlabel=None,
    xlabel_fontsize=26,
    ylabel="Count",
    ylabel_fontsize=26,

    # === Axis ===
    xlim=None,
    ylim=None,
    xtick_fontsize=22,
    ytick_fontsize=22,
    spine_linewidth=1.6,
    integer_yticks=True,

    # === Grid ===
    grid=True,
    grid_style="--",
    grid_alpha=0.4,

    # === Output ===
    out_path=None,
    show=False,
):
    """
    VisCoreスタイルのヒストグラム描画関数（KDE対応版）
    """

    # フォント
    setup_fonts()

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # ------------------------------
    #  ビン計算
    # ------------------------------
    if isinstance(bins, str):
        computed_bins = _calc_bins(data, bins)
    else:
        computed_bins = bins

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------
    #  スタイル設定
    # ------------------------------
    for s in ax.spines.values():
        s.set_linewidth(spine_linewidth)

    # ------------------------------
    #  ヒストグラム本体
    # ------------------------------
    ax.hist(
        data,
        bins=computed_bins,
        range=range,
        color=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
    )

    # ------------------------------
    #  KDE（核密度推定）
    # ------------------------------
    if kde:
        kde_model = gaussian_kde(data)

        xs = np.linspace(
            xlim[0] if xlim else data.min(),
            xlim[1] if xlim else data.max(),
            400
        )
        ys = kde_model(xs)

        # 正規化：ヒストグラムとスケールを合わせる
        bin_width = computed_bins[1] - computed_bins[0]
        ys_scaled = ys * len(data) * bin_width

        ax.plot(
            xs,
            ys_scaled,
            color=kde_color,
            linewidth=kde_linewidth,
            linestyle=kde_linestyle,
        )

    # ------------------------------
    #  Labels / Titles
    # ------------------------------
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=10)

    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=10)

    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=12)

    # ------------------------------
    #  Axis limits
    # ------------------------------
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.tick_params(axis="x", labelsize=xtick_fontsize, direction="in", pad=10)
    ax.tick_params(axis="y", labelsize=ytick_fontsize, direction="in", pad=10)

    if integer_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ------------------------------
    # Grid
    # ------------------------------
    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    # ------------------------------
    # Save / Show
    # ------------------------------
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

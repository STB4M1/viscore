# viscore/plotting_2d/lineplot_2d.py

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
from pathlib import Path


def create_lineplot_2d(
    data_dict,
    x_col,
    y_col,
    output_image_path,

    # =========================
    # ラベル・タイトル
    # =========================
    xlabel="",
    ylabel="",
    title=None,

    xlabel_fontsize=22,
    ylabel_fontsize=22,
    title_fontsize=22,

    # =========================
    # 図全体
    # =========================
    figsize=(8.5, 6),
    dpi=200,

    # =========================
    # 線・マーカー
    # =========================
    marker="o",
    markersize=4,
    linewidth=1.8,
    linestyle="-",
    alpha=1.0,

    # =========================
    # 軸範囲
    # =========================
    x_lim=None,
    y_lim=None,

    # =========================
    # tick 配置
    # =========================
    x_integer=False,
    x_tick_interval=None,
    y_tick_interval=None,

    # =========================
    # tick 表示形式
    # =========================
    x_tick_format=None,
    y_tick_format=None,

    # =========================
    # pad
    # =========================
    xlabel_pad=5,
    ylabel_pad=10,
    tick_pad=5,

    # =========================
    # tick 見た目
    # =========================
    tick_labelsize=18,
    tick_direction="in",

    # =========================
    # grid
    # =========================
    grid=True,
    grid_style="--",
    grid_alpha=0.4,

    # =========================
    # spine
    # =========================
    spine_width=1.5,

    # =========================
    # 色指定
    # =========================
    color_dict=None,
    colors=None,
    colormap=None,
    color_cycle_start=0,

    # =========================
    # 凡例制御
    # =========================
    show_legend=True,
    legend_outside=True,
    legend_loc="center left",
    legend_bbox=(1.02, 0.5),
    legend_fontsize=14,
    legend_ncol=1,
    legend_every=1,
    legend_max_items=None,
    legend_formatter=None,

    # =========================
    # その他
    # =========================
    sort_labels=True,
):
    """
    VisCore: 汎用 2D 折れ線プロット（完全制御版）

    data_dict : dict[key -> DataFrame]
      - key は ID / 状態 / 物理量名など
      - 表示は legend_formatter に委ねる
    """

    # =========================
    # Figure
    # =========================
    plt.figure(figsize=figsize)

    # =========================
    # ラベル順
    # =========================
    labels = list(data_dict.keys())
    if sort_labels:
        labels = sorted(labels)

    # =========================
    # 色決定
    # =========================
    if color_dict is not None:
        color_list = [color_dict.get(label, None) for label in labels]

    elif colors is not None:
        if len(colors) < len(labels):
            raise ValueError("colors length < number of datasets")
        color_list = colors

    elif colormap is not None:
        cmap = cm.get_cmap(colormap, len(labels) + color_cycle_start)
        color_list = [cmap(i + color_cycle_start) for i in range(len(labels))]

    else:
        color_list = [None] * len(labels)

    # =========================
    # 凡例表示判定
    # =========================
    def use_in_legend(i):
        if legend_max_items is not None and i >= legend_max_items:
            return False
        if legend_every > 1 and i % legend_every != 0:
            return False
        return True

    # =========================
    # Plot
    # =========================
    for i, (key, color) in enumerate(zip(labels, color_list)):
        df = data_dict[key]

        disp_label = (
            legend_formatter(key)
            if legend_formatter is not None
            else str(key)
        )

        plt.plot(
            df[x_col],
            df[y_col],
            label=disp_label if use_in_legend(i) else "_nolegend_",
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )

    # =========================
    # Labels / Title
    # =========================
    plt.xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=xlabel_pad)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=ylabel_pad)

    if title is not None:
        plt.title(title, fontsize=title_fontsize)

    # =========================
    # Axis & ticks
    # =========================
    ax = plt.gca()

    if x_integer:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if x_tick_interval is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))
    if y_tick_interval is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))

    if x_tick_format is not None:
        ax.xaxis.set_major_formatter(FormatStrFormatter(x_tick_format))
    if y_tick_format is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))

    if x_lim is not None:
        plt.xlim(*x_lim)
    if y_lim is not None:
        plt.ylim(*y_lim)

    plt.tick_params(
        axis="both",
        labelsize=tick_labelsize,
        direction=tick_direction,
        pad=tick_pad,
    )

    # =========================
    # Grid
    # =========================
    if grid:
        plt.grid(True, linestyle=grid_style, alpha=grid_alpha)

    # =========================
    # Spine
    # =========================
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    # =========================
    # Legend
    # =========================
    if show_legend:
        plt.legend(
            loc=legend_loc if legend_outside else None,
            bbox_to_anchor=legend_bbox if legend_outside else None,
            fontsize=legend_fontsize,
            frameon=False,
            ncol=legend_ncol,
        )

    # =========================
    # Save
    # =========================
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, bbox_inches="tight", dpi=dpi)
    plt.close()

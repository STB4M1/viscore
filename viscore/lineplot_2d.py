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
    # 右軸（追加）
    # =========================
    use_right_axis=False,              # Trueで右軸を有効化
    right_axis_keys=None,              # 右軸に載せるdata_dictのkey一覧（list/set/tuple）
    ylabel_right="",                   # 右軸ラベル
    y2_lim=None,                       # 右軸のylim
    y2_tick_interval=None,             # 右軸tick間隔
    y2_tick_format=None,               # 右軸tick表示フォーマット（例 "%.2f"）

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
    linestyle_dict=None,   # dict[label -> linestyle]
    linestyles=None,       # list/tuple of linestyles (labels順)

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
    VisCore: 汎用 2D 折れ線プロット（右軸オプション対応）
    data_dict : dict[key -> DataFrame]
    """

    # =========================
    # Figure / Axes
    # =========================
    fig, ax1 = plt.subplots(figsize=figsize)

    ax2 = None
    right_key_set = set(right_axis_keys) if (use_right_axis and right_axis_keys is not None) else set()
    if use_right_axis:
        ax2 = ax1.twinx()

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
    # 線種決定
    # =========================
    if linestyle_dict is not None:
        linestyle_list = [linestyle_dict.get(label, linestyle) for label in labels]
    elif linestyles is not None:
        if len(linestyles) < len(labels):
            raise ValueError("linestyles length < number of datasets")
        linestyle_list = list(linestyles)
    else:
        linestyle_list = [linestyle] * len(labels)

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
    plotted_count = 0
    for key, color, ls_i in zip(labels, color_list, linestyle_list):
        df = data_dict[key]
        disp_label = legend_formatter(key) if legend_formatter is not None else str(key)

        # 右軸に載せるか判定
        target_ax = ax1
        if use_right_axis and (key in right_key_set):
            target_ax = ax2

        target_ax.plot(
            df[x_col],
            df[y_col],
            label=disp_label if use_in_legend(plotted_count) else "_nolegend_",
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            linestyle=ls_i,
            alpha=alpha,
        )
        plotted_count += 1

    # =========================
    # Labels / Title
    # =========================
    ax1.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=xlabel_pad)
    ax1.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=ylabel_pad)

    if use_right_axis:
        ax2.set_ylabel(ylabel_right, fontsize=ylabel_fontsize, labelpad=ylabel_pad)

    if title is not None:
        ax1.set_title(title, fontsize=title_fontsize)

    # =========================
    # Axis & ticks (xは共通)
    # =========================
    if x_integer:
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if x_tick_interval is not None:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))

    if x_tick_format is not None:
        ax1.xaxis.set_major_formatter(FormatStrFormatter(x_tick_format))

    # 左y
    if y_tick_interval is not None:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))
    if y_tick_format is not None:
        ax1.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format))
    if y_lim is not None:
        ax1.set_ylim(*y_lim)

    # 右y
    if use_right_axis:
        if y2_tick_interval is not None:
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(y2_tick_interval))
        if y2_tick_format is not None:
            ax2.yaxis.set_major_formatter(FormatStrFormatter(y2_tick_format))
        if y2_lim is not None:
            ax2.set_ylim(*y2_lim)

    if x_lim is not None:
        ax1.set_xlim(*x_lim)

    # tick params（左右両方に適用）
    ax1.tick_params(axis="both", labelsize=tick_labelsize, direction=tick_direction, pad=tick_pad)
    if use_right_axis:
        ax2.tick_params(axis="y", labelsize=tick_labelsize, direction=tick_direction, pad=tick_pad)

    # =========================
    # Grid（左軸に対して）
    # =========================
    if grid:
        ax1.grid(True, linestyle=grid_style, alpha=grid_alpha)

    # =========================
    # Spine
    # =========================
    for spine in ax1.spines.values():
        spine.set_linewidth(spine_width)
    if use_right_axis:
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)

    # =========================
    # Legend（左右を結合）
    # =========================
    if show_legend:
        handles1, labels1 = ax1.get_legend_handles_labels()
        if use_right_axis:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2
            lbls = labels1 + labels2
        else:
            handles = handles1
            lbls = labels1

        fig.legend(
            handles,
            lbls,
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

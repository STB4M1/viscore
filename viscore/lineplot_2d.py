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

    # =========================
    # 右 y 軸（secondary axis）
    # =========================
    use_secondary_y=False,
    secondary_y_keys=None,          # iterable of keys to plot on right y-axis
    secondary_ylabel=None,
    secondary_y_lim=None,
    secondary_y_tick_interval=None,
    secondary_y_tick_format=None,

    vlines=None,   # e.g. [(123.4, dict(linestyle="--", linewidth=2, alpha=0.8))]
    hlines=None,
):
    """
    VisCore: 汎用 2D 折れ線プロット（完全制御版）

    data_dict : dict[key -> DataFrame]
      - key は ID / 状態 / 物理量名など
      - 表示は legend_formatter に委ねる

    右 y 軸:
      use_secondary_y = True のときのみ有効。
      secondary_y_keys に含まれる key の系列は右 y 軸に描画される。
    """

    # =========================
    # Figure
    # =========================
    plt.figure(figsize=figsize)

    # =========================
    # Axes
    # =========================
    ax = plt.gca()
    ax2 = None
    secondary_key_set = set(secondary_y_keys) if (use_secondary_y and secondary_y_keys is not None) else set()

    if use_secondary_y:
        ax2 = ax.twinx()

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
        # ラベルごとの指定。指定が無いラベルはデフォルト linestyle を使う
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
    for i, (key, color, ls_i) in enumerate(zip(labels, color_list, linestyle_list)):
        df = data_dict[key]

        disp_label = (
            legend_formatter(key)
            if legend_formatter is not None
            else str(key)
        )

        # 右軸に乗せるキーかどうか
        target_ax = ax2 if (ax2 is not None and key in secondary_key_set) else ax

        target_ax.plot(
            df[x_col],
            df[y_col],
            label=disp_label if use_in_legend(i) else "_nolegend_",
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            linestyle=ls_i,
            alpha=alpha,
        )

    # =========================
    # Labels / Title
    # =========================
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=xlabel_pad)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=ylabel_pad)

    if ax2 is not None and secondary_ylabel is not None:
        ax2.set_ylabel(secondary_ylabel, fontsize=ylabel_fontsize, labelpad=ylabel_pad)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    # =========================
    # Axis & ticks (primary)
    # =========================
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
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # =========================
    # Axis & ticks (secondary y)
    # =========================
    if ax2 is not None:
        if secondary_y_tick_interval is not None:
            ax2.yaxis.set_major_locator(
                ticker.MultipleLocator(secondary_y_tick_interval)
            )
        if secondary_y_tick_format is not None:
            ax2.yaxis.set_major_formatter(
                FormatStrFormatter(secondary_y_tick_format)
            )
        if secondary_y_lim is not None:
            ax2.set_ylim(*secondary_y_lim)

    # =========================
    # Tick 見た目
    # =========================
    ax.tick_params(
        axis="both",
        labelsize=tick_labelsize,
        direction=tick_direction,
        pad=tick_pad,
    )
    if ax2 is not None:
        ax2.tick_params(
            axis="y",
            labelsize=tick_labelsize,
            direction=tick_direction,
            pad=tick_pad,
        )

    # =========================
    # Grid
    # =========================
    if grid:
        # 元コードでは plt.grid だったが、ax が gca() なので挙動は同じ
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    # =========================
    # Extra lines (vline / hline)
    # =========================
    if vlines is not None:
        for item in vlines:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
                x, kw = item
                ax.axvline(x, **kw)
            else:
                ax.axvline(item)

    if hlines is not None:
        for item in hlines:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
                y, kw = item
                ax.axhline(y, **kw)
            else:
                ax.axhline(item)

    # =========================
    # Spine
    # =========================
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if ax2 is not None:
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)

    # =========================
    # Legend
    # =========================
    if show_legend:
        if use_secondary_y and ax2 is not None:
            # 両軸の line を統合
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2
            legend_labels = labels1 + labels2

            plt.legend(
                handles,
                legend_labels,
                loc=legend_loc if legend_outside else None,
                bbox_to_anchor=legend_bbox if legend_outside else None,
                fontsize=legend_fontsize,
                frameon=False,
                ncol=legend_ncol,
            )
        else:
            # === 元コードと完全に同じ呼び出し ===
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

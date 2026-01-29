import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
from pathlib import Path
import numpy as np


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

    marker_dict=None,      # dict[label -> marker]
    markers=None,          # list/tuple markers (labels順)
    markersize_dict=None,  # dict[label -> markersize]
    markersizes=None,      # list/tuple markersizes (labels順)
    linewidth_dict=None,   # dict[label -> linewidth]
    linewidths=None,       # list/tuple linewidths (labels順)

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

    # =========================
    # scale (linear / log / symlog / logit)
    # =========================
    x_scale="linear",
    y_scale="linear",
    secondary_y_scale="linear",

    # =========================
    # minor tick（副目盛）primary
    # =========================
    primary_show_minor_ticks=None,
    primary_x_minor_tick_interval=None,   # Noneなら自動
    primary_y_minor_tick_interval=None,   # Noneなら自動
    primary_major_tick_length=None,
    primary_minor_tick_length=None,

    # =========================
    # minor tick（副目盛）secondary (right y)
    # =========================
    secondary_show_minor_ticks=None,
    secondary_y_minor_tick_interval=None, # Noneなら自動
    secondary_major_tick_length=None,
    secondary_minor_tick_length=None,

    # =========================
    # Errorbar 
    # =========================
    use_errorbar=None,              # None: 自動判定, True/False: 強制
    yerr_col=None,                  # symmetric: ±err
    yerr_low_col=None,              # asymmetric lower (bound or delta)
    yerr_high_col=None,             # asymmetric upper (bound or delta)
    yerr_bounds_are_absolute=True,  # True: low/high are bounds, False: low/high are deltas
    capsize=3,
    elinewidth=1.2,
    ecolor=None,
    errorbar_alpha=None,

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
    # marker 決定
    # =========================
    if marker_dict is not None:
        marker_list = [marker_dict.get(label, marker) for label in labels]
    elif markers is not None:
        if len(markers) < len(labels):
            raise ValueError("markers length < number of datasets")
        marker_list = list(markers)
    else:
        marker_list = [marker] * len(labels)

    # =========================
    # markersize 決定
    # =========================
    if markersize_dict is not None:
        markersize_list = [markersize_dict.get(label, markersize) for label in labels]
    elif markersizes is not None:
        if len(markersizes) < len(labels):
            raise ValueError("markersizes length < number of datasets")
        markersize_list = list(markersizes)
    else:
        markersize_list = [markersize] * len(labels)

    # =========================
    # linewidth 決定
    # =========================
    if linewidth_dict is not None:
        linewidth_list = [linewidth_dict.get(label, linewidth) for label in labels]
    elif linewidths is not None:
        if len(linewidths) < len(labels):
            raise ValueError("linewidths length < number of datasets")
        linewidth_list = list(linewidths)
    else:
        linewidth_list = [linewidth] * len(labels)

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
    for i, (key, color, ls_i, mk_i, ms_i, lw_i) in enumerate(
        zip(labels, color_list, linestyle_list, marker_list, markersize_list, linewidth_list)
    ):
        df = data_dict[key]

        disp_label = (
            legend_formatter(key)
            if legend_formatter is not None
            else str(key)
        )

        target_ax = ax2 if (ax2 is not None and key in secondary_key_set) else ax

        # --- errorbar を使うか判定（後方互換）---
        has_yerr = (yerr_col is not None) or (yerr_low_col is not None and yerr_high_col is not None)
        do_errorbar = has_yerr if (use_errorbar is None) else bool(use_errorbar)

        if do_errorbar:
            # --- yerr を構築 ---
            y = df[y_col].to_numpy()

            yerr = None
            if yerr_col is not None:
                # symmetric: ±err
                yerr = df[yerr_col].to_numpy()
            elif (yerr_low_col is not None) and (yerr_high_col is not None):
                low = df[yerr_low_col].to_numpy()
                high = df[yerr_high_col].to_numpy()
                if yerr_bounds_are_absolute:
                    # low/high は絶対境界：y - low, high - y
                    yerr = np.vstack([y - low, high - y])
                else:
                    # low/high はdelta（幅）として扱う
                    yerr = np.vstack([low, high])

            target_ax.errorbar(
                df[x_col],
                y,
                yerr=yerr,
                label=disp_label if use_in_legend(i) else "_nolegend_",
                color=color,
                marker=mk_i,
                markersize=ms_i,
                linewidth=lw_i,
                linestyle=ls_i,
                alpha=alpha if errorbar_alpha is None else errorbar_alpha,
                capsize=capsize,
                elinewidth=elinewidth,
                ecolor=ecolor,
            )
        else:
            # 既存どおり
            target_ax.plot(
                df[x_col],
                df[y_col],
                label=disp_label if use_in_legend(i) else "_nolegend_",
                color=color,
                marker=mk_i,
                markersize=ms_i,
                linewidth=lw_i,
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
    # Scale (primary)
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)

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


    # =========================================================
    # Guarantee major ticks when x_tick_interval is specified
    # (No new args, keep backward behavior unless x_lim is also set)
    # =========================================================
    if (x_tick_interval is not None) and (x_lim is not None) and (x_scale in (None, "linear")):
        xmin, xmax = x_lim
        # 浮動小数誤差対策で少しだけ余裕を持たせる
        eps = 1e-9 * max(1.0, abs(xmax - xmin))
        start = np.ceil((xmin - eps) / x_tick_interval) * x_tick_interval
        end   = np.floor((xmax + eps) / x_tick_interval) * x_tick_interval
        ticks = np.arange(start, end + x_tick_interval * 0.5, x_tick_interval)
        ax.set_xticks(ticks)

    # =========================
    # Axis & ticks (secondary y)
    # =========================
    # Scale (secondary y)
    if ax2 is not None and secondary_y_scale is not None:
        ax2.set_yscale(secondary_y_scale)
        
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
    # Minor ticks (primary)  ※Noneなら触らない（後方互換）
    # =========================
    if primary_show_minor_ticks is True:
        ax.minorticks_on()
    elif primary_show_minor_ticks is False:
        ax.minorticks_off()

    # interval 固定は「副目盛をONにする指定がある時だけ」＋「指定がある時だけ」
    if primary_show_minor_ticks is True:
        if primary_x_minor_tick_interval is not None and (x_scale in (None, "linear")):
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(primary_x_minor_tick_interval))
        if primary_y_minor_tick_interval is not None and (y_scale in (None, "linear")):
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(primary_y_minor_tick_interval))


    # =========================
    # Minor ticks (secondary y) ※Noneなら触らない（後方互換）
    # =========================
    if ax2 is not None:
        if secondary_show_minor_ticks is True:
            ax2.minorticks_on()
        elif secondary_show_minor_ticks is False:
            ax2.minorticks_off()

        if secondary_show_minor_ticks is True:
            if secondary_y_minor_tick_interval is not None and (secondary_y_scale in (None, "linear")):
                ax2.yaxis.set_minor_locator(ticker.MultipleLocator(secondary_y_minor_tick_interval))

    # =========================
    # Tick 見た目（backward compatible）
    # =========================
    # primary major（lengthは指定されたときだけ）
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_labelsize,
        direction=tick_direction,
        pad=tick_pad,
        **({} if primary_major_tick_length is None else {"length": primary_major_tick_length}),
    )

    # primary minor（lengthは指定されたときだけ）
    ax.tick_params(
        axis="both",
        which="minor",
        direction=tick_direction,
        pad=tick_pad,
        **({} if primary_minor_tick_length is None else {"length": primary_minor_tick_length}),
    )

    # secondary y
    if ax2 is not None:
        ax2.tick_params(
            axis="y",
            which="major",
            labelsize=tick_labelsize,
            direction=tick_direction,
            pad=tick_pad,
            **({} if secondary_major_tick_length is None else {"length": secondary_major_tick_length}),
        )
        ax2.tick_params(
            axis="y",
            which="minor",
            direction=tick_direction,
            pad=tick_pad,
            **({} if secondary_minor_tick_length is None else {"length": secondary_minor_tick_length}),
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

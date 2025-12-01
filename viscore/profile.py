import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_plot_profile(
    measured,                           # pd.DataFrame or list[pd.DataFrame]
    theoretical=None,                   # pd.DataFrame or list[pd.DataFrame]
    output_profile_path="",

    # --- cross-section settings ---
    x=None, y=None,
    x_start=None, x_end=None,
    y_start=None, y_end=None,

    # --- pixel pitch ---
    x_pixel_pitch=None,
    y_pixel_pitch=None,

    # --- figure settings ---
    fig_size=(6, 4),
    dpi_val=200,

    # --- measured style ---
    measured_colors=None,
    measured_alphas=None,
    measured_labels=None,

    # --- theoretical style ---
    theoretical_colors=None,
    theoretical_ls=None,
    theoretical_alphas=None,
    theoretical_labels=None,

    # --- tick interval ---
    x_tick_interval=None,
    y_tick_interval=None,

    # --- reference lines ---
    reference_lines=None,
    reference_line_labels=None,
    reference_line_color="gray",
    reference_line_ls=":",
    reference_line_width=0.8,

    # --- axis limits ---
    y_limits=None,

    # --- axis labels ---
    xlabel_str="$x$ [μm]",
    ylabel_str="$y$ [μm]",
    phase_label="Phase [rad]",

    xlabel_font_size=20,
    ylabel_font_size=20,
    phase_label_font_size=20,

    tick_label_size=16,
    tick_direction="in",

    # --- axis frame (spines) ---
    spine_linewidth=1.5,

    # --- legend ---
    legend_font_size=14,
    legend_loc="upper center",
    legend_anchor=(0.5, 1.15),
    legend_ncol=3,
    legend_frame=False,

    # --- title ---
    title=None,
    title_font_size=18,

    # --- filename rules ---
    filename_suffix_x="_x_{x}_y_{y_start}-{y_end}.png",
    filename_suffix_y="_y_{y}_x_{x_start}-{x_end}.png",

    # --- preprocessing ---
    invert_y=True,
    col_x="x",
    col_y="y",
    col_phase="phase",
):
    """
    measured / theoretical は DataFrame または list[DataFrame]。
    基準線 reference_lines は任意のリストを指定可能。
    ラベル類すべて外部指定に対応した完全カスタム関数。
    """

    # ============================================================
    # 入力の統一
    # ============================================================
    if isinstance(measured, pd.DataFrame):
        measured = [measured]
    if theoretical is None:
        theoretical = []
    elif isinstance(theoretical, pd.DataFrame):
        theoretical = [theoretical]

    # measured デフォルト色など
    if measured_colors is None:
        measured_colors = ["royalblue"] * len(measured)
    if measured_alphas is None:
        measured_alphas = [1.0] * len(measured)
    if measured_labels is None:
        measured_labels = [f"Measured {i+1}" for i in range(len(measured))]

    # theoretical デフォルト
    if theoretical_colors is None:
        theoretical_colors = ["black"] * len(theoretical)
    if theoretical_ls is None:
        theoretical_ls = ["--"] * len(theoretical)
    if theoretical_alphas is None:
        theoretical_alphas = [1.0] * len(theoretical)
    if theoretical_labels is None:
        theoretical_labels = [f"Theoretical {i+1}" for i in range(len(theoretical))]

    # reference line labels
    if reference_lines is not None and reference_line_labels is None:
        reference_line_labels = [None] * len(reference_lines)

    # ============================================================
    # 前処理
    # ============================================================
    def preprocess(df):
        if df is None:
            return None
        df = df.rename(columns={col_x: "x", col_y: "y", col_phase: "phase"})
        if invert_y:
            df["y"] = df["y"].max() - df["y"]
        return df

    measured = [preprocess(df) for df in measured]
    theoretical = [preprocess(df) for df in theoretical]

    # ============================================================
    # 抽出関数
    # ============================================================
    def extract(df, fix_x=None, fix_y=None):
        if df is None:
            return None
        if fix_x is not None:
            return df[(df["x"] == fix_x) & (df["y"] >= y_start) & (df["y"] <= y_end)]
        if fix_y is not None:
            return df[(df["y"] == fix_y) & (df["x"] >= x_start) & (df["x"] <= x_end)]
        return None

    # ============================================================
    # 描画本体
    # ============================================================
    def plot_profile(direction="x"):
        plt.figure(figsize=fig_size)
        ax = plt.gca()

        # --- reference lines ---
        if reference_lines is not None:
            for ref, label in zip(reference_lines, reference_line_labels):
                ax.axhline(ref, color=reference_line_color,
                           linestyle=reference_line_ls,
                           linewidth=reference_line_width,
                           label=label)

        # --- theoretical ---
        for df, col, ls, alp, lbl in zip(
            theoretical, theoretical_colors, theoretical_ls, theoretical_alphas, theoretical_labels
        ):
            prof = extract(df, fix_x=x if direction == "x" else None,
                              fix_y=y if direction == "y" else None)
            if prof is not None and not prof.empty:
                xx = prof["y"] * y_pixel_pitch if direction == "x" else prof["x"] * x_pixel_pitch
                ax.plot(xx, prof["phase"], color=col, linestyle=ls, alpha=alp, label=lbl)

        # --- measured ---
        for df, col, alp, lbl in zip(measured, measured_colors, measured_alphas, measured_labels):
            prof = extract(df, fix_x=x if direction == "x" else None,
                              fix_y=y if direction == "y" else None)
            if prof is not None and not prof.empty:
                xx = prof["y"] * y_pixel_pitch if direction == "x" else prof["x"] * x_pixel_pitch
                ax.plot(xx, prof["phase"], color=col, alpha=alp, label=lbl)

        # ====================================================
        # 軸ラベル
        # ====================================================
        if direction == "x":
            ax.xlabel = ylabel_str
            plt.xlabel(ylabel_str, fontsize=xlabel_font_size)
        else:
            plt.xlabel(xlabel_str, fontsize=xlabel_font_size)

        plt.ylabel(phase_label, fontsize=phase_label_font_size)

        # ====================================================
        # tick interval
        # ====================================================
        if direction == "x" and y_tick_interval is not None:
            xmin, xmax = ax.get_xlim()
            ax.set_xticks(np.arange(xmin, xmax + 1e-6, y_tick_interval))
        elif direction == "y" and x_tick_interval is not None:
            xmin, xmax = ax.get_xlim()
            ax.set_xticks(np.arange(xmin, xmax + 1e-6, x_tick_interval))

        plt.tick_params(axis="both", labelsize=tick_label_size, direction=tick_direction)
        plt.grid(True)

        # ====================================================
        # axis limits
        # ====================================================
        if y_limits is not None:
            plt.ylim(y_limits)

        # ====================================================
        # frame
        # ====================================================
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)

        # ====================================================
        # legend
        # ====================================================
        ax.legend(
            fontsize=legend_font_size,
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            ncol=legend_ncol,
            frameon=legend_frame
        )

        # ====================================================
        # title
        # ====================================================
        if title is not None:
            plt.title(title, fontsize=title_font_size)

        # ====================================================
        # save file
        # ====================================================
        Path(output_profile_path).parent.mkdir(parents=True, exist_ok=True)

        if direction == "x":
            suffix = filename_suffix_x.format(x=x, y_start=y_start, y_end=y_end)
        else:
            suffix = filename_suffix_y.format(y=y, x_start=x_start, x_end=x_end)

        plt.savefig(output_profile_path.replace(".png", suffix),
                    dpi=dpi_val, bbox_inches="tight")
        plt.close()

    # ============================================================
    # 実行
    # ============================================================
    if x is not None:
        plot_profile("x")

    if y is not None:
        plot_profile("y")

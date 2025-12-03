import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
from pathlib import Path


def create_vector_field_2d(
    x, y, dx, dy,
    output_image_path,

    # --- preprocessing ---
    swap_xy=False,
    invert_y=True,
    normalize_origin=False,

    # --- figure / appearance ---
    fig_size=(6, 3),
    dpi_val=200,
    aspect_equal=True,

    # --- pixel pitch ---
    x_pixel_pitch=None,
    y_pixel_pitch=None,

    # --- color / colormap ---
    colormap="viridis",
    cbar_range=None,
    cbar_num_ticks=4,
    cbar_label=r"$|v|$ [px$/$frame]",
    cbar_font_size=20,
    cbar_ticks_label_size=16,
    cbar_label_pad=8,
    cbar_fraction=0.08,
    cbar_pad=0.05,
    cbar_shrink=1.0,

    # --- tick settings ---
    tick_label_size=16,
    tick_direction='in',
    x_tick_pad=5,
    y_tick_pad=5,

    # --- log scale options ---
    cbar_log=False,

    # --- axis labels ---
    x_label_str=r"$x$ [μm]",
    y_label_str=r"$y$ [μm]",
    x_label_font_size=20,
    y_label_font_size=20,
    x_label_pad=10,
    y_label_pad=10,

    # --- axis frame (spines) ---
    spine_linewidth=1.5,

    # --- title ---
    title=None,
    title_font_size=12,
    title_pad=6,

    # --- arrow settings ---
    arrow_scale=1.0,
    arrow_linewidth=0.8, 
    arrow_width=0.004,   
):
    """
    Julia PIV の .dat を Python で正しく再現するための
    ベクトル場可視化関数。
    """

    # ============================================================
    # 1. reshape（ここが最重要！！！）
    # ============================================================
    xs = np.unique(x)
    ys = np.unique(y)
    nx = len(xs)
    ny = len(ys)

    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    DX = dx.reshape(ny, nx)
    DY = dy.reshape(ny, nx)

    # ============================================================
    # 2. preprocessing
    # ============================================================
    if swap_xy:
        X, Y = Y.copy(), X.copy()
        DX, DY = DY.copy(), DX.copy()

    if invert_y:
        Y = Y.max() - Y
        DY = -DY

    if normalize_origin:
        X = X - X.min()
        Y = Y - Y.min()

    if x_pixel_pitch is not None:
        X = X * x_pixel_pitch
    if y_pixel_pitch is not None:
        Y = Y * y_pixel_pitch

    # magnitude
    V = np.sqrt(DX**2 + DY**2)

    # ============================================================
    # 3. color range
    # ============================================================
    vmin = float(np.nanmin(V))
    vmax = float(np.nanmax(V))
    if cbar_range is None:
        cmin, cmax = vmin, vmax
    else:
        cmin, cmax = cbar_range

    # ============================================================
    # 4. figure
    # ============================================================
    fig, ax = plt.subplots(figsize=fig_size)
    norm = mcolors.LogNorm(vmin=cmin, vmax=cmax) if cbar_log else mcolors.Normalize(vmin=cmin, vmax=cmax)

    q = ax.quiver(
        X, Y,
        DX * arrow_scale, DY * arrow_scale,
        V,
        cmap=colormap,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=1,
        linewidth=arrow_linewidth,   
        width=arrow_width,           
    )

    # ============================================================
    # 5. colorbar
    # ============================================================
    cbar = plt.colorbar(q, ax=ax, fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)
    ticks = np.geomspace(cmin, cmax, cbar_num_ticks) if cbar_log else np.linspace(cmin, cmax, cbar_num_ticks)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=cbar_ticks_label_size)
    cbar.set_label(cbar_label, fontsize=cbar_font_size, labelpad=cbar_label_pad)

    # ============================================================
    # 6. axis settings
    # ============================================================
    if aspect_equal:
        ax.set_aspect("equal")

    ax.set_xlabel(x_label_str, fontsize=x_label_font_size, labelpad=x_label_pad)
    ax.set_ylabel(y_label_str, fontsize=y_label_font_size, labelpad=y_label_pad)

    ax.tick_params(axis='x', direction=tick_direction, labelsize=tick_label_size, pad=x_tick_pad)
    ax.tick_params(axis='y', direction=tick_direction, labelsize=tick_label_size, pad=y_tick_pad)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)

    if title is not None:
        ax.set_title(title, fontsize=title_font_size, pad=title_pad)

    # ============================================================
    # 7. save
    # ============================================================
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=dpi_val, bbox_inches="tight")
    plt.close(fig)

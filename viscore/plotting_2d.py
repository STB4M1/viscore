import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

def create_heatmap_2d(
    data,
    output_image_path,

    # --- data preprocessing ---
    invert_y=True,
    normalize_origin=True,

    # --- figure / appearance ---
    fig_size=(6, 3),
    dpi_val=200,
    aspect_equal=True,

    # --- cropping ---
    x_min=None, x_max=None,
    y_min=None, y_max=None,
    z_min=None, z_max=None,

    # --- pixel pitch ---
    x_pixel_pitch=None,
    y_pixel_pitch=None,

    # --- colormap / colorbar ---
    colormap=None,
    cbar_range=None,
    cbar_num_ticks=4,
    cbar_label="Phase [rad]",
    cbar_font_size=20,
    cbar_ticks_label_size=16,
    cbar_label_pad=8,
    cbar_fraction=0.08,
    cbar_pad=0.05,
    cbar_shrink=0.8,

    # --- tick settings ---
    x_tick_interval=None,
    y_tick_interval=None,
    tick_label_size=16,
    tick_direction='in',
    x_tick_pad=0,
    y_tick_pad=0,

    # --- tick format (decimal places) ---
    x_tick_decimals=None,
    y_tick_decimals=None,
    cbar_tick_decimals=None,

    # --- log scale options ---
    x_log=False,
    y_log=False,
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
):
    # ============================================================
    # 1. Preprocessing
    # ============================================================
    data = data.rename(columns={'x': 'y_temp', 'y': 'x'}).rename(columns={'y_temp': 'y'})

    if invert_y:
        data['y'] = data['y'].max() - data['y']

    if normalize_origin:
        data['x'] -= data['x'].min()
        data['y'] -= data['y'].min()

    # ============================================================
    # 2. Cropping
    # ============================================================
    if (x_min is not None) and (x_max is not None):
        data = data[(data['x'] >= x_min) & (data['x'] <= x_max)]

    if (y_min is not None) and (y_max is not None):
        data = data[(data['y'] >= y_min) & (data['y'] <= y_max)]

    # ============================================================
    # 3. Pivot → coordinate scaling
    # ============================================================
    pivot_data = data.pivot(index='y', columns='x', values='phase')

    if x_pixel_pitch is None or y_pixel_pitch is None:
        raise ValueError("x_pixel_pitch と y_pixel_pitch を指定してください")

    x = pivot_data.columns.values * x_pixel_pitch
    y = pivot_data.index.values * y_pixel_pitch
    Z = pivot_data.values

    # ============================================================
    # 4. Color range
    # ============================================================
    zmin_data = np.nanmin(Z)
    zmax_data = np.nanmax(Z)

    if np.isclose(zmin_data, zmax_data):
        eps = 1e-6 if zmin_data == 0 else abs(zmin_data)*1e-6
        zmin_data -= eps
        zmax_data += eps

    cmin = cbar_range[0] if cbar_range is not None else (z_min if z_min is not None else zmin_data)
    cmax = cbar_range[1] if cbar_range is not None else (z_max if z_max is not None else zmax_data)

    # ============================================================
    # 5. Figure
    # ============================================================
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(
        Z,
        cmap=colormap,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=cmin,
        vmax=cmax,
        aspect='equal' if aspect_equal else None,
    )

    # ============================================================
    # 5.1 Log scale for axes
    # ============================================================
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    # ============================================================
    # 6. Colorbar
    # ============================================================
    # log カラーバーのバリデーション
    if cbar_log and cmin <= 0:
        raise ValueError("cbar_log=True のときは cmin > 0 である必要があります。")

    if cbar_log:
        norm = mcolors.LogNorm(vmin=cmin, vmax=cmax)
        im.set_norm(norm)
        ticks = np.geomspace(cmin, cmax, cbar_num_ticks)
    else:
        ticks = np.linspace(cmin, cmax, cbar_num_ticks)

    cbar = plt.colorbar(
        im,
        ax=ax,
        fraction=cbar_fraction,
        pad=cbar_pad,
        shrink=cbar_shrink,
        ticks=ticks
    )

    # カラーバー小数点フォーマット
    if cbar_tick_decimals is not None:
        fmt = "%." + str(cbar_tick_decimals) + "f"
        cbar.formatter = FormatStrFormatter(fmt)
        cbar.update_ticks()

    cbar.set_label(cbar_label, fontsize=cbar_font_size)
    cbar.ax.tick_params(labelsize=cbar_ticks_label_size)
    cbar.ax.yaxis.labelpad = cbar_label_pad

    # ============================================================
    # 7. Axis ticks / labels
    # ============================================================
    if x_tick_interval is not None:
        ax.set_xticks(np.arange(x.min(), x.max() + 1e-9, x_tick_interval))

    if y_tick_interval is not None:
        ax.set_yticks(np.arange(y.min(), y.max() + 1e-9, y_tick_interval))

    ax.set_xlabel(x_label_str, fontsize=x_label_font_size, labelpad=x_label_pad)
    ax.set_ylabel(y_label_str, fontsize=y_label_font_size, labelpad=y_label_pad)

    ax.tick_params(axis='x', labelsize=tick_label_size, direction=tick_direction, pad=x_tick_pad)
    ax.tick_params(axis='y', labelsize=tick_label_size, direction=tick_direction, pad=y_tick_pad)

    # ============================================================
    # 7.1 Decimal formatting for axis ticks
    # ============================================================
    def apply_tick_format(axis, decimals):
        if decimals is not None:
            fmt = "%." + str(decimals) + "f"
            axis.set_major_formatter(FormatStrFormatter(fmt))

    apply_tick_format(ax.xaxis, x_tick_decimals)
    apply_tick_format(ax.yaxis, y_tick_decimals)

    # ============================================================
    # 8. Spine linewidth
    # ============================================================
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)

    # ============================================================
    # 9. Title
    # ============================================================
    if title is not None:
        ax.set_title(title, fontsize=title_font_size, pad=title_pad)

    # ============================================================
    # 10. Save
    # ============================================================
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(output_image_path, dpi=dpi_val, bbox_inches='tight')
    finally:
        plt.close(fig)

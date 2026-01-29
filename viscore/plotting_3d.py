import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

def create_plot_3d(
    data,
    output_image_path,

    # --- data preprocessing ---
    swap_xy=True,
    invert_y=True,

    # --- figure / appearance ---
    fig_size=(7, 3),
    dpi_val=200,
    box_aspect_z_scale=0.5,
    box_aspect=None,

    # --- cropping / range settings ---
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
    cbar_boundaries_num=100,
    cbar_label="Phase [rad]",
    cbar_font_size=12,
    cbar_ticks_label_size=9,
    cbar_fraction=0.06,
    cbar_pad=0.05,
    cbar_shrink=0.8,

    # --- axis labels ---
    x_label_str=r"$x$ [μm]",
    y_label_str=r"$y$ [μm]",
    z_label_str="",
    x_label_font_size=12,
    y_label_font_size=12,
    z_label_font_size=None,
    x_label_pad=8,
    y_label_pad=1,
    z_label_pad=0,

    # --- tick settings ---
    x_tick_interval=None,
    y_tick_interval=None, 
    x_ticks_label_size=9,
    y_ticks_label_size=9,
    z_ticks_label_size=10,
    x_ticks_pad=0,
    y_ticks_pad=0,
    z_ticks_pad=0,

    # --- tick format (decimal places) ---
    x_tick_decimals=None,
    y_tick_decimals=None,
    z_tick_decimals=None,
    cbar_tick_decimals=None,

    # --- log scale options ---
    x_log=False,
    y_log=False,
    z_log=False,
    cbar_log=False,

    # --- camera / view ---
    view_elev=30,
    view_azim=240,

    # --- surface plot resolution ---
    rcount=None,
    ccount=None,

    # --- title ---
    title=None,
    title_font_size=12,
    title_pad=6,

    # --- save options ---
    use_tight=True,

    # --- NEW: margins (0〜1の正規化値) ---
    margin_left=None,
    margin_right=None,
    margin_top=None,
    margin_bottom=None,
):
    # ============================================================
    # 1. Data preprocessing
    # ============================================================
    if swap_xy:
        data = data.rename(columns={'x': 'y_temp', 'y': 'x'}).rename(columns={'y_temp': 'y'})

    # y 軸反転
    if invert_y:
        data['y'] = data['y'].max() - data['y']

    # ============================================================
    # 2. Data cropping
    # ============================================================
    if (x_min is not None) and (x_max is not None):
        data = data[(data['x'] >= x_min) & (data['x'] <= x_max)]

    if (y_min is not None) and (y_max is not None):
        data = data[(data['y'] >= y_min) & (data['y'] <= y_max)]

    # pivot → meshgrid
    pivot_data = data.pivot(index='y', columns='x', values='phase')

    if x_pixel_pitch is None or y_pixel_pitch is None:
        raise ValueError("x_pixel_pitch と y_pixel_pitch を指定してください")
    
    x = pivot_data.columns.values * x_pixel_pitch
    y = pivot_data.index.values * y_pixel_pitch

    X, Y = np.meshgrid(x, y)
    Z = pivot_data.values

    # ============================================================
    # 3. Determine Z-range / colorbar range
    # ============================================================
    zmin_data = np.nanmin(Z)
    zmax_data = np.nanmax(Z)

    # すべて同値なら見た目用に幅を作る
    if np.isclose(zmin_data, zmax_data):
        eps = 1e-6 if zmin_data == 0 else abs(zmin_data) * 1e-6
        zmin_data -= eps
        zmax_data += eps

    # カラーバーのレンジ決定
    cmin = (
        cbar_range[0] if cbar_range is not None
        else (z_min if z_min is not None else zmin_data)
    )
    cmax = (
        cbar_range[1] if cbar_range is not None
        else (z_max if z_max is not None else zmax_data)
    )

    # ============================================================
    # 4. Figure / axes 作成
    # ============================================================
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # 描画範囲
    x_lo, x_hi = x.min(), x.max()
    y_lo, y_hi = y.min(), y.max()

    ax.set_xlim(x_lo, x_hi, auto=False)
    ax.set_ylim(y_lo, y_hi, auto=False)

    # Z-range 設定
    if (z_min is not None) and (z_max is not None):
        ax.set_zlim(z_min, z_max)
    else:
        z_min, z_max = ax.get_zlim()

    # 軸の範囲
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo

    # aspect 設定（完全指定が最優先）
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)
    else:
        ax.set_box_aspect((x_range, y_range, x_range * box_aspect_z_scale))
        
    # ============================================================
    # 5. Tick settings
    # ============================================================
    def make_ticks(lo, hi, step):
        start = np.floor(lo / step) * step
        return np.arange(start, hi + 1e-9, step)

    if x_tick_interval is not None:
        ax.set_xticks(make_ticks(x_lo, x_hi, x_tick_interval))
    if y_tick_interval is not None:
        ax.set_yticks(make_ticks(y_lo, y_hi, y_tick_interval))

    if rcount is None:
        rcount = 200
    if ccount is None:
        ccount = 200

    # ============================================================
    # 5.1 Axis scale (log)
    # ============================================================
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if z_log:
        ax.set_zscale("log")

    # ============================================================
    # 5.2 Tick decimal formatting
    # ============================================================
    def apply_tick_format(axis, decimals):
        if decimals is not None:
            fmt = "%." + str(decimals) + "f"
            axis.set_major_formatter(FormatStrFormatter(fmt))

    apply_tick_format(ax.xaxis, x_tick_decimals)
    apply_tick_format(ax.yaxis, y_tick_decimals)
    apply_tick_format(ax.zaxis, z_tick_decimals)

    # ============================================================
    # 6. Surface plot
    # ============================================================
    norm = mcolors.LogNorm(vmin=cmin, vmax=cmax) if cbar_log else None

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=colormap,
        norm=norm,
        vmin=None if cbar_log else cmin,
        vmax=None if cbar_log else cmax,
        rcount=rcount,
        ccount=ccount,
    )

    ax.view_init(elev=view_elev, azim=view_azim)

    # ============================================================
    # 7. Colorbar
    # ============================================================
    ticks = np.linspace(cmin, cmax, cbar_num_ticks)

    if cbar_log:
        ticks = np.geomspace(cmin, cmax, cbar_num_ticks)  # 対数目盛
    else:
        ticks = np.linspace(cmin, cmax, cbar_num_ticks)

    cbar = fig.colorbar(
        surf,
        ax=ax,
        orientation='vertical',
        fraction=cbar_fraction,
        pad=cbar_pad,
        shrink=cbar_shrink,
        ticks=ticks
    )

    # 小数点桁数指定
    if cbar_tick_decimals is not None:
        fmt = "%." + str(cbar_tick_decimals) + "f"
        cbar.formatter = FormatStrFormatter(fmt)
        cbar.update_ticks()


    cbar.set_label(cbar_label, fontsize=cbar_font_size)
    cbar.ax.tick_params(labelsize=cbar_ticks_label_size)

    # ============================================================
    # 8. Axes labels / ticks / title
    # ============================================================
    ax.set_xlabel(x_label_str, fontsize=x_label_font_size, labelpad=x_label_pad)
    ax.set_ylabel(y_label_str, fontsize=y_label_font_size, labelpad=y_label_pad)
    ax.set_zlabel(z_label_str, fontsize=z_label_font_size, labelpad=z_label_pad)

    ax.tick_params(axis='x', labelsize=x_ticks_label_size, pad=x_ticks_pad)
    ax.tick_params(axis='y', labelsize=y_ticks_label_size, pad=y_ticks_pad)
    ax.tick_params(axis='z', labelsize=z_ticks_label_size, pad=z_ticks_pad)

    if title is not None:
        ax.set_title(title, fontsize=title_font_size, pad=title_pad)

    # 8.5 Adjust margins
    if any(v is not None for v in [margin_left, margin_right, margin_top, margin_bottom]):
        fig.subplots_adjust(
            left=margin_left if margin_left is not None else None,
            right=margin_right if margin_right is not None else None,
            top=margin_top if margin_top is not None else None,
            bottom=margin_bottom if margin_bottom is not None else None
        )

    # ============================================================
    # 9. Save
    # ============================================================
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        bbox_option = "tight" if use_tight else None
        plt.savefig(output_image_path, dpi=dpi_val, bbox_inches=bbox_option)
    finally:
        plt.close(fig)



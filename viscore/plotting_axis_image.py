import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt


def add_axis_to_image(
    in_img_path,
    output_image_path,

    # --- image processing ---
    brightness=1.0,
    contrast=1.0,

    # --- pixel pitch ---
    pitch_x=None,
    pitch_y=None,

    # --- figure settings ---
    fig_size=(10, 6),
    dpi_val=200,
    cmap="gray",

    # --- NEW: percentile display range (完全後方互換) ---
    use_percentile=False,          # Falseなら従来通り vmin=0,vmax=255
    p_low=1.0,                     # use_percentile=True のときのみ有効
    p_high=99.0,                   # use_percentile=True のときのみ有効

    # --- NEW: gamma correction (完全後方互換) ---
    gamma=None,                    # Noneなら従来通り（gamma補正なし）

    # --- time / title ---
    frame=None,
    frame_list=None,
    fps=200,
    show_time_title=True,
    title_font_size=22,
    title_pad=12,

    # --- axis labels ---
    x_label=r"$x$ [μm]",
    y_label=r"$y$ [μm]",
    label_font_size=26,
    label_pad=10,

    # --- ticks ---
    tick_size=20,
    tick_direction='in',
    tick_length=5,
    tick_width=1.2,
    tick_pad=8,

    x_tick_interval=500,
    y_tick_interval=500,

    # --- axis frame ---
    spine_width=2.0,
):
    """
    画像を読み込み、明るさ・コントラスト調整したうえで、
    ピクセルピッチに基づく座標軸を付けて保存する関数。

    ✅ 完全後方互換:
      - use_percentile=False (default) → vmin=0,vmax=255 のまま
      - gamma=None (default) → gamma補正なし
    """

    # ------------------------------------------------------------
    # 1. Load image & preprocess
    # ------------------------------------------------------------
    if pitch_x is None or pitch_y is None:
        raise ValueError("pitch_x と pitch_y を指定してください。")

    img = Image.open(in_img_path)

    # 既存処理（後方互換維持）
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    img_array = np.array(img)

    # --- NEW: gamma correction (optional) ---
    # gamma < 1.0 → 明るく、gamma > 1.0 → 暗く
    if gamma is not None:
        if gamma <= 0:
            raise ValueError("gamma は正の値にしてください（例: 0.7, 1.2）。")

        # uint8 想定の LUT で高速に処理（RGB/Gray どちらもOK）
        arr = img_array.astype(np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0) ** gamma
        img_array = (arr * 255.0 + 0.5).astype(np.uint8)

    height, width = img_array.shape[:2]

    # ------------------------------------------------------------
    # 2. Create figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)

    # --- NEW: percentile-based display range (optional) ---
    if use_percentile:
        # RGBなら輝度に落としてpercentileを決める（表示自体は元配列のまま）
        if img_array.ndim == 3:
            # ITU-R BT.601 に近い係数（見た目安定）
            lum = 0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2]
            vmin = float(np.percentile(lum, p_low))
            vmax = float(np.percentile(lum, p_high))
        else:
            vmin = float(np.percentile(img_array, p_low))
            vmax = float(np.percentile(img_array, p_high))

        # 同値などで壊れないように保険
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 255.0
    else:
        # 従来挙動
        vmin, vmax = 0, 255

    ax.imshow(
        img_array,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[0, width * pitch_x, 0, height * pitch_y],
    )

    # ------------------------------------------------------------
    # 3. Title (convert frame → time option)
    # ------------------------------------------------------------
    if show_time_title and frame is not None and frame_list is not None:
        if frame in frame_list:
            idx = frame_list.index(frame)
            t = idx / fps
            ax.set_title(f"$t = {t:.3f}\ \mathrm{{s}}$", fontsize=title_font_size, pad=title_pad)
        else:
            ax.set_title(f"frame = {frame}", fontsize=title_font_size, pad=title_pad)

    # ------------------------------------------------------------
    # 4. Labels
    # ------------------------------------------------------------
    ax.set_xlabel(x_label, fontsize=label_font_size, labelpad=label_pad)
    ax.set_ylabel(y_label, fontsize=label_font_size, labelpad=label_pad)

    # ------------------------------------------------------------
    # 5. Ticks
    # ------------------------------------------------------------
    if x_tick_interval is not None:
        xticks = np.arange(0, width * pitch_x + 1e-9, x_tick_interval)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"${int(t)}$" for t in xticks])

    if y_tick_interval is not None:
        yticks = np.arange(0, height * pitch_y + 1e-9, y_tick_interval)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"${int(t)}$" for t in yticks])

    ax.tick_params(
        direction=tick_direction,
        length=tick_length,
        width=tick_width,
        pad=tick_pad,
        labelsize=tick_size,
    )

    # ------------------------------------------------------------
    # 6. Axis frame (spines)
    # ------------------------------------------------------------
    for s in ax.spines.values():
        s.set_linewidth(spine_width)

    # ------------------------------------------------------------
    # 7. Save figure
    # ------------------------------------------------------------
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=dpi_val, bbox_inches='tight')
    finally:
        plt.close(fig)

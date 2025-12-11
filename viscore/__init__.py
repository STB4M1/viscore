# viscore/__init__.py

# --- 初期化処理 ---
from .styles.tex_fonts import setup_fonts
from .styles.colormaps import register_cmthermal

setup_fonts()
register_cmthermal()

# --- 関数を公開 ---
from .plotting_2d import create_heatmap_2d
from .plotting_vector_2d import create_vector_field_2d
from .plotting_3d import create_plot_3d
from .profile import create_plot_profile
from .plotting_axis_image import add_axis_to_image

__all__ = [
    "create_heatmap_2d",
    "create_vector_field_2d",
    "create_plot_3d",
    "create_plot_profile",
    "add_axis_to_image"
]

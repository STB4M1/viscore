# viscore/styles/colormaps.py

import matplotlib as mpl
import matplotlib.colors as mcolors

# -----------------------------------------------------------
# 1) cmthermal（元の名前・配色そのまま）
# -----------------------------------------------------------
cmthermal = mcolors.LinearSegmentedColormap.from_list(
    "cmthermal",
    ["#1c3f75", "#068fb9", "#f1e235", "#d64e8b", "#730e22"],
    256
)

# -----------------------------------------------------------
# 2) NEW: azure_sunset（青 → 明るい → オレンジ）
# -----------------------------------------------------------
azure_sunset = mcolors.LinearSegmentedColormap.from_list(
    "viscore.azure_sunset",
    ["#1E3A8A", "#38BDF8", "#E0F2FE", "#FDE68A", "#F97316"],
    256
)

# -----------------------------------------------------------
# 3) NEW: aurora（青 → 緑（真ん中） → オレンジ）
# -----------------------------------------------------------
aurora = mcolors.LinearSegmentedColormap.from_list(
    "viscore.aurora",
    ["#1E3A8A", "#38BDF8", "#86EFAC", "#FDE68A", "#F97316"],
    256
)

# -----------------------------------------------------------
# 4) 登録ヘルパ
# -----------------------------------------------------------
def _register(cmap, name: str):
    try:
        mpl.colormaps.register(cmap, name=name, override_builtin=True)
    except Exception:
        pass

def register_cmthermal():
    _register(cmthermal, "viscore.cmthermal")

def register_azure_sunset():
    _register(azure_sunset, "viscore.azure_sunset")

def register_aurora():
    _register(aurora, "viscore.aurora")

def register_all_colormaps():
    register_cmthermal()
    register_azure_sunset()
    register_aurora()

# -----------------------------------------------------------
# 5) 外部公開
# -----------------------------------------------------------
__all__ = [
    "cmthermal",
    "azure_sunset",
    "aurora",
    "register_cmthermal",
    "register_azure_sunset",
    "register_aurora",
    "register_all_colormaps",
]

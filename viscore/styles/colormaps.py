# viscore/styles/colormaps.py

import matplotlib as mpl
import matplotlib.colors as mcolors

# -----------------------------------------------------------
# ★ 1. cmthermal をグローバルに定義（import 可能にする）
# -----------------------------------------------------------
cmthermal = mcolors.LinearSegmentedColormap.from_list(
    "cmthermal",
    ["#1c3f75", "#068fb9", "#f1e235", "#d64e8b", "#730e22"],
    256
)

# -----------------------------------------------------------
# ★ 2. matplotlib に登録するだけの関数
# -----------------------------------------------------------
def register_cmthermal():
    try:
        mpl.colormaps.register(cmthermal, name="cmthermal", override_builtin=True)
    except Exception:
        pass

# -----------------------------------------------------------
# ★ 3. 外部公開
# -----------------------------------------------------------
__all__ = ["cmthermal", "register_cmthermal"]

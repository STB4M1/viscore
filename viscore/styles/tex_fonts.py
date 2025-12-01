import matplotlib as mpl
from matplotlib import font_manager

def setup_fonts():
    font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
    try:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        mpl.rcParams.update({
            "font.family": font_name,
            "mathtext.fontset": "cm",
        })
    except Exception:
        pass

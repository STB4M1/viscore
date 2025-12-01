# VisCore – 科学・工学データのための 2D/3D 可視化ツールキット

**VisCore** は、科学・工学・シミュレーション・画像計測など、  
幅広い数値データをきれいに可視化するために設計している **軽量 Python 可視化ライブラリ** です。

2D ヒートマップ、3D サーフェスプロット、プロファイル図など、  
研究用途の図を「シンプルなコード」で「高品質・統一デザイン」で生成できます。

---

## ✨ 主な特徴

### 🔹 2D ヒートマップ
- ピクセルピッチに基づく座標スケーリング  
- カラーバー、範囲指定、軸設定などを柔軟にカスタム  
- 論文・発表資料向けの高品質画像を簡単生成

### 🔹 3D サーフェスプロット
- 視点角度・アスペクト比・Zレンジなどを細かく設定可能  
- 計測データ・シミュレーション出力を直感的に可視化  

### 🔹 統一スタイル
- Times フォント自動登録（環境にある場合）  
- 自作カラーマップ **cmthermal** を標準搭載  
- 毎回同じスタイルで図を作れるため、研究資料に最適

### 🔹 シンプルな API
- Pandas DataFrame を渡すだけの簡潔な設計  
- バッチ処理・自動化スクリプトとも相性抜群  

---

## 🚀 使用例

### ■ 3D プロット

```python
from viscore import create_plot_3d, cmthermal
import pandas as pd

df = pd.read_csv("phase_data.dat", sep=r"\s+", names=["x", "y", "phase"])

create_plot_3d(
    data=df,
    output_image_path="output_3d.png",
    x_pixel_pitch=6.9,
    y_pixel_pitch=6.9,
    colormap=cmthermal,
)
```

---

### ■ 2D ヒートマップ

```python
from viscore import create_heatmap_2d, cmthermal

create_heatmap_2d(
    data=df,
    output_image_path="output_2d.svg",
    x_pixel_pitch=6.9,
    y_pixel_pitch=6.9,
    colormap=cmthermal,
)
```

---

## 📁 プロジェクト構成

```
viscore/
 ├── __init__.py
 ├── plotting_2d.py
 ├── plotting_3d.py
 ├── profile.py
 └── styles/
      ├── fonts.py
      └── colormaps.py
```

---

## 🎨 スタイル / カラーマップ

Viscore では以下が自動的に適用されます：

- Times フォント（存在する場合）  
- Thermal カラーマップ `cmthermal`（独自実装）  

独自スタイルを追加したい場合は  
`viscore/styles/` を拡張するだけでOKです。


---

## 📜 ライセンス

MIT License

---

## 👤 作者

Author: Mitsuki ISHIYAMA  

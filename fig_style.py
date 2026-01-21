import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
# 全体の設定
plt.rcParams['font.family'] = 'serif'  # セリフ体フォントを使用 (論文の本文と合わせるため)
plt.rcParams['font.serif'] = ['Times New Roman', 'Times'] # 使用するセリフ体フォント
plt.rcParams['font.size'] = 18           # 基本フォントサイズ (本文のフォントサイズに合わせる)
plt.rcParams['axes.labelsize'] = 18      # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 15     # x軸目盛ラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 15     # y軸目盛ラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 18     # 凡例のフォントサイズ
plt.rcParams['figure.dpi'] = 600         # ディスプレイ上のDPI (保存時は別途指定する場合が多い)
plt.rcParams['savefig.dpi'] = 600        # 保存時のDPI (出版社の要件に合わせる: 300-600dpi)

# 線の太さや表示形式に関する設定
plt.rcParams['lines.linewidth'] = 1.5    # 線の太さ
plt.rcParams['axes.linewidth'] = 0.5     # 軸線の太さ
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.direction'] = 'in'   # 目盛線を内側に向ける
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True         # グリッド線を表示 (必要に応じて)
plt.rcParams['grid.alpha'] = 0.5         # グリッド線の透明度

# PDF保存時の設定 (Type 1フォント埋め込み推奨)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 凡例の設定
plt.rcParams['legend.fancybox'] = False   # 凡例の枠を角丸にしない
plt.rcParams['legend.edgecolor'] = 'black' # 凡例の枠線の色
plt.rcParams['legend.framealpha'] = 1.0   # 凡例の背景の透明度 (1.0で不透明)

# レイアウト調整
plt.rcParams['figure.constrained_layout.use'] = True # 自動レイアウト調整
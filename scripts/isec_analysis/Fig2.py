import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize
# 全体の設定
plt.rcParams['font.family'] = 'serif'  # セリフ体フォントを使用 (論文の本文と合わせるため)
plt.rcParams['font.serif'] = ['Times New Roman', 'Times'] # 使用するセリフ体フォント
plt.rcParams['font.size'] = 18           # 基本フォントサイズ (本文のフォントサイズに合わせる)
plt.rcParams['axes.labelsize'] = 18      # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 15     # x軸目盛ラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 15     # y軸目盛ラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 18     # 凡例のフォントサイズ
plt.rcParams['figure.dpi'] = 300         # ディスプレイ上のDPI (保存時は別途指定する場合が多い)
plt.rcParams['savefig.dpi'] = 300        # 保存時のDPI (出版社の要件に合わせる: 300-600dpi)

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
# 共通データをインポート
from exp.iseC_affine.common_data import (
    V_t,
    V_0,
    V_ext,
    analytes,
    log_M,
    V_R_data,
)

# K_avの計算
def calculate_K_av(V_R, V_0, V_t):
    return (V_R - V_0) / (V_t - V_0)

# 各環境のK_avを計算
K_av_data = {}
for env, V_R in V_R_data.items():
    K_av_data[env] = calculate_K_av(V_R, V_0, V_t)

# 基準環境と基準データ
baseline_env = 's2_66ACN'
K_av_baseline = K_av_data[baseline_env]

def calculate_optimal_shift(K_av_target, K_av_baseline):
    """
    Compute the vertical shift that minimizes RSS
    (residual sum of squares).
    """
    def rss(shift):
        return np.sum((K_av_target + shift - K_av_baseline)**2)

    result = minimize(rss, x0=0.0, method='BFGS')
    optimal_shift = result.x[0]
    min_rss = result.fun
    return optimal_shift, min_rss

# Compute optimal shifts for each condition
shifts = {}
rss_values = {}
shifted_K_av = {}

for env in K_av_data:
    if env == baseline_env:
        shifts[env] = 0.0
        rss_values[env] = 0.0
        shifted_K_av[env] = K_av_data[env]
    else:
        shift, rss = calculate_optimal_shift(K_av_data[env], K_av_baseline)
        shifts[env] = shift
        rss_values[env] = rss
        shifted_K_av[env] = K_av_data[env] + shift

# Create plots (2x2 layout)
fig = plt.figure(figsize=(14, 10))

# 環境ごとの色とマーカー
colors = {'s1_60ACN': '#1f77b4', 's2_66ACN': '#ff7f0e', 
          's3_70ACN': '#2ca02c', 's4_75ACN': '#d62728'}
markers = {'s1_60ACN': 'o', 's2_66ACN': 's', 
           's3_70ACN': '^', 's4_75ACN': 'D'}
labels = {'s1_60ACN': '60% ACN', 's2_66ACN': '66% ACN (baseline)', 
          's3_70ACN': '70% ACN', 's4_75ACN': '75% ACN'}

# (a) Overlay after shifting
ax1 = plt.subplot(2, 2, 1)
for env in ['s1_60ACN', 's2_66ACN', 's3_70ACN', 's4_75ACN']:
    ax1.plot(log_M, shifted_K_av[env], 
            marker=markers[env], 
            color=colors[env],
            label=labels[env],
            linewidth=2,
            markersize=7,
            alpha=0.8)

ax1.set_xlabel('log M', fontsize=11, fontweight='bold')
ax1.set_ylabel('K_av (shifted)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Vertical Shift to Baseline', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=9)
ax1.set_ylim(0, 1.1)

# (b) Residuals plot (difference from baseline)
ax2 = plt.subplot(2, 2, 2)
for env in ['s1_60ACN', 's3_70ACN', 's4_75ACN']:  # 基準を除く
    residuals = shifted_K_av[env] - K_av_baseline
    ax2.plot(log_M, residuals, 
            marker=markers[env], 
            color=colors[env],
            label=labels[env],
            linewidth=2,
            markersize=7,
            alpha=0.8)

ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('log M', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residual (shifted - baseline)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Residuals After Shift', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=9)

# (c) Before vs After comparison (example: s1_60ACN)
ax3 = plt.subplot(2, 2, 3)
compare_env = 's1_60ACN'
ax3.plot(log_M, K_av_data[compare_env], 
        marker='o', color='gray', label='Original (60% ACN)',
        linewidth=2, markersize=7, alpha=0.6, linestyle='--')
ax3.plot(log_M, shifted_K_av[compare_env], 
        marker='o', color=colors[compare_env], label='Shifted (60% ACN)',
        linewidth=2, markersize=7, alpha=0.8)
ax3.plot(log_M, K_av_baseline, 
        marker='s', color=colors[baseline_env], label='Baseline (66% ACN)',
        linewidth=2, markersize=7, alpha=0.8)

ax3.set_xlabel('log M', fontsize=11, fontweight='bold')
ax3.set_ylabel('K_av', fontsize=11, fontweight='bold')
ax3.set_title('(c) Example: 60% ACN Before/After Shift', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='best', fontsize=9)
ax3.set_ylim(0, 1.1)

# (d) Bar plot of RSS values
ax4 = plt.subplot(2, 2, 4)
env_names = [labels[env] for env in ['s1_60ACN', 's3_70ACN', 's4_75ACN']]
rss_vals = [rss_values[env] for env in ['s1_60ACN', 's3_70ACN', 's4_75ACN']]
bar_colors = [colors[env] for env in ['s1_60ACN', 's3_70ACN', 's4_75ACN']]

bars = ax4.bar(env_names, rss_vals, color=bar_colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('RSS (Residual Sum of Squares)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Goodness of Fit (RSS)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

# RSS値をバーの上に表示
for bar, val in zip(bars, rss_vals):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Figure 2: Vertical Shift Alignment to Baseline Condition (s2_66ACN)\nC12 Polymer Monolith Column', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('exp/iseC_affine/K_av_vertical_shift_analysis.png', dpi=300, bbox_inches='tight')

# Results summary output
print("=== Vertical Shift Analysis Summary ===")
print(f"\nBaseline condition: {labels[baseline_env]}")
print(f"\nOptimal shift values (to align with baseline):")
for env in ['s1_60ACN', 's3_70ACN', 's4_75ACN']:
    print(f"  {labels[env]}: shift = {shifts[env]:+.4f}, RSS = {rss_values[env]:.6f}")

print(f"\nMeasurement noise level estimate:")
print(f"  Mean RSS: {np.mean([rss_values[env] for env in ['s1_60ACN', 's3_70ACN', 's4_75ACN']]):.6f}")
print(f"  Number of data points: {len(log_M)}")

# Residual statistics after shifting
print(f"\nResidual statistics (after shift):")
for env in ['s1_60ACN', 's3_70ACN', 's4_75ACN']:
    residuals = shifted_K_av[env] - K_av_baseline
    print(f"  {labels[env]}:")
    print(f"    Mean: {np.mean(residuals):+.6f}")
    print(f"    Std:  {np.std(residuals):.6f}")
    print(f"    Max absolute: {np.max(np.abs(residuals)):.6f}")
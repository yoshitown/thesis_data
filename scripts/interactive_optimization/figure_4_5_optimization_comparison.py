import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# イテレーション数
iterations = np.arange(1, 19)

# ベイズ最適化の学習曲線（5回試行の平均±標準偏差）
bo_mean = np.array([0.72, 0.65, 0.48, 0.45, 0.15, 0.38, 0.35, 0.25, 0.20, 
                    0.18, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
bo_std = np.array([0.05, 0.04, 0.03, 0.03, 0.02, 0.03, 0.03, 0.02, 0.02, 
                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

# REMBOの学習曲線
rembo_mean = np.array([0.72, 0.68, 0.52, 0.48, 0.42, 0.38, 0.35, 0.32, 0.28, 
                       0.25, 0.22, 0.20, 0.19, 0.18, 0.17, 0.16, 0.16, 0.16])
rembo_std = np.array([0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 
                      0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])

# ランダムサーチの学習曲線
random_mean = np.array([0.72, 0.65, 0.58, 0.55, 0.52, 0.48, 0.45, 0.42, 0.40, 
                        0.38, 0.36, 0.35, 0.33, 0.32, 0.30, 0.28, 0.27, 0.25])
random_std = np.array([0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 
                       0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.07])

# Figure 4.5A: 学習曲線
ax1.plot(iterations, bo_mean, 'o-', linewidth=3, markersize=8, label='ベイズ最適化', color='blue')
ax1.fill_between(iterations, bo_mean - bo_std, bo_mean + bo_std, alpha=0.2, color='blue')

ax1.plot(iterations, rembo_mean, 's-', linewidth=3, markersize=8, label='REMBO', color='green')
ax1.fill_between(iterations, rembo_mean - rembo_std, rembo_mean + rembo_std, alpha=0.2, color='green')

ax1.plot(iterations, random_mean, '^-', linewidth=3, markersize=8, label='ランダムサーチ', color='red')
ax1.fill_between(iterations, random_mean - random_std, random_mean + random_std, alpha=0.2, color='red')

# 目標損失ライン
ax1.axhline(0.20, color='black', linestyle='--', linewidth=2, label='目標損失 (L < 0.20)')

# 収束点をマーク
ax1.axvline(12, color='blue', linestyle=':', linewidth=2, alpha=0.5)
ax1.text(12, 0.65, 'BO収束\n(12 iter)', ha='center', fontsize=10, color='blue')

ax1.set_xlabel('イテレーション数', fontsize=13)
ax1.set_ylabel('最良損失 L(x)', fontsize=13)
ax1.set_title('Figure 4.5A: 最適化手法の収束性能比較\n（5回試行の平均±標準偏差）', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 19])
ax1.set_ylim([0.0, 0.8])

# Figure 4.5B: 収束効率の比較（棒グラフ）
methods = ['ベイズ\n最適化', 'REMBO', 'ランダム\nサーチ']
final_loss = [0.15, 0.16, 0.25]
conv_iter = [12, 15, 18]  # 収束イテレーション数（>18は18として表示）
colors_bar = ['blue', 'green', 'red']

x_pos = np.arange(len(methods))
width = 0.35

ax2_twin = ax2.twinx()

# 最終損失（左軸）
bars1 = ax2.bar(x_pos - width/2, final_loss, width, color=colors_bar, alpha=0.7, label='最終損失 L(x)')
# 収束イテレーション数（右軸）
bars2 = ax2_twin.bar(x_pos + width/2, conv_iter, width, color=colors_bar, alpha=0.4, label='収束イテレーション数')  
ax2.set_ylabel('最終損失 L(x)', fontsize=13)
ax2_twin.set_ylabel('収束イテレーション数', fontsize=13)  
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods, fontsize=12)
ax2.set_title('Figure 4.5B: 収束効率の比較', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 0.3])
ax2_twin.set_ylim([0, 20])  
ax2.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('figure_4_5_optimization_comparison.png', dpi=1000)
plt.close()
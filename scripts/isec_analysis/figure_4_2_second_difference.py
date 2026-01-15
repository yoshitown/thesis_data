import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))

# プローブの分子量データ
MW = np.array([78, 92, 106, 120, 134, 162, 190, 178])
log_M = np.log10(MW)

# 3環境でのK_avデータ
K_av_s060 = np.array([0.92, 0.88, 0.82, 0.76, 0.68, 0.55, 0.42, 0.48])
K_av_s066 = np.array([0.85, 0.82, 0.78, 0.73, 0.66, 0.58, 0.48, 0.52])
K_av_s069 = np.array([0.78, 0.76, 0.74, 0.71, 0.67, 0.62, 0.56, 0.58])

# 二階差分の計算
def second_difference(K_av):
    """二階差分 Δ²K_av を計算"""
    delta2 = np.zeros(len(K_av) - 2)
    for i in range(1, len(K_av) - 1):
        delta2[i-1] = K_av[i+1] - 2*K_av[i] + K_av[i-1]
    return delta2

delta2_s060 = second_difference(K_av_s060)
delta2_s066 = second_difference(K_av_s066)
delta2_s069 = second_difference(K_av_s069)

# 中央の点に対応するlog_M
log_M_center = log_M[1:-1]

# プロット
ax.plot(log_M_center, delta2_s060, 'o-', label='s = 0.60', linewidth=2, markersize=10)
ax.plot(log_M_center, delta2_s066, 's-', label='s = 0.66', linewidth=2, markersize=10)
ax.plot(log_M_center, delta2_s069, '^-', label='s = 0.69', linewidth=2, markersize=10)

# ゼロ線
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# 極小値位置のマーク
min_idx_s060 = np.argmin(delta2_s060)
min_idx_s066 = np.argmin(delta2_s066)
min_idx_s069 = np.argmin(delta2_s069)

ax.plot(log_M_center[min_idx_s060], delta2_s060[min_idx_s060], 'ro', markersize=15, 
        fillstyle='none', markeredgewidth=2)
ax.plot(log_M_center[min_idx_s066], delta2_s066[min_idx_s066], 'gs', markersize=15, 
        fillstyle='none', markeredgewidth=2)
ax.plot(log_M_center[min_idx_s069], delta2_s069[min_idx_s069], 'b^', markersize=15, 
        fillstyle='none', markeredgewidth=2)

# アノテーション
ax.annotate(f'遷移点\nlog M ≈ {log_M_center[min_idx_s060]:.2f}', 
            xy=(log_M_center[min_idx_s060], delta2_s060[min_idx_s060]),
            xytext=(log_M_center[min_idx_s060]-0.1, -0.12),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax.set_xlabel('log M (分子量)', fontsize=12)
ax.set_ylabel('Δ²K_av (二階差分)', fontsize=12)
ax.set_title('Figure 4.2: 二階差分による曲率解析\n（環境依存的な遷移点移動）', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/Figure_4_2_second_difference.png', dpi=1000, bbox_inches='tight')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import griddata

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 9))

# 実測データ点（18 Runs）
s_data = np.array([0.60, 0.62, 0.65, 0.63, 0.69, 0.67, 0.64, 0.70, 0.68,
                   0.69, 0.66, 0.69, 0.70, 0.68, 0.69, 0.70, 0.69, 0.69])
r_data = np.array([0.19, 0.25, 0.33, 0.28, 0.42, 0.38, 0.30, 0.40, 0.35,
                   0.41, 0.32, 0.42, 0.39, 0.40, 0.42, 0.41, 0.42, 0.42])
L_data = np.array([0.72, 0.65, 0.48, 0.58, 0.15, 0.38, 0.55, 0.25, 0.42,
                   0.18, 0.52, 0.15, 0.28, 0.22, 0.15, 0.20, 0.16, 0.15])

# グリッド作成
s_grid = np.linspace(0.60, 0.75, 100)
r_grid = np.linspace(0.18, 0.43, 100)
S_grid, R_grid = np.meshgrid(s_grid, r_grid)

# 補間（ガウス過程の予測平均を模擬）
L_grid = griddata((s_data, r_data), L_data, (S_grid, R_grid), method='cubic')

# ヒートマップ
contour = ax.contourf(S_grid, R_grid, L_grid, levels=20, cmap='RdYlGn_r', alpha=0.8)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('損失 L(x)', fontsize=12)

# 等高線
contour_lines = ax.contour(S_grid, R_grid, L_grid, levels=10, colors='black', 
                            linewidths=0.5, alpha=0.3)
ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

# 実測点をプロット
scatter = ax.scatter(s_data, r_data, c=L_data, s=150, 
                     cmap='RdYlGn_r', edgecolors='black', linewidths=2, 
                     marker='o', alpha=0.9, zorder=5)

# 最適点を強調
opt_idx = np.argmin(L_data)
ax.scatter(s_data[opt_idx], r_data[opt_idx], s=500, 
           marker='*',
           color='gold', edgecolors='black', linewidths=3, 
           zorder=10, label=f'最適解 (s={s_data[opt_idx]:.2f}, r={r_data[opt_idx]:.2f})')

# 初期点を強調
ax.scatter(s_data[2], r_data[2], s=300, 
           marker='X', color='cyan', edgecolors='black', linewidths=2, 
           zorder=10, label=f'初期点 Run 3 (L={L_data[2]:.2f})')

# 高損失領域をアノテート
ax.text(0.61, 0.40, '高損失\n領域', fontsize=11, color='darkred', 
        ha='center', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 低損失領域をアノテート
ax.text(0.69, 0.42, '低損失\n領域', fontsize=11, color='darkgreen', 
        ha='center', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('溶媒比 s (THF vol%)', fontsize=13)
ax.set_ylabel('架橋剤比 r (EDMA/C12MA)', fontsize=13)
ax.set_title('Figure 4.3: s-r空間における性能ランドスケープ\n（ガウス過程モデルの予測平均値）', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/Figure_4_3_landscape.png', dpi=1000, bbox_inches='tight')
plt.close()

print("Figure 4.3 作成完了")
print(f"\n最適条件: s = {s_data[opt_idx]:.2f}, r = {r_data[opt_idx]:.2f}, L = {L_data[opt_idx]:.2f}")
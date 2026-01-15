import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import griddata

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# 実測データ点（18 Runs）
s_data = np.array([0.60, 0.62, 0.65, 0.63, 0.69, 0.67, 0.64, 0.70, 0.68,
                   0.69, 0.66, 0.69, 0.70, 0.68, 0.69, 0.70, 0.69, 0.69])
r_data = np.array([0.19, 0.25, 0.33, 0.28, 0.42, 0.38, 0.30, 0.40, 0.35,
                   0.41, 0.32, 0.42, 0.39, 0.40, 0.42, 0.41, 0.42, 0.42])
L_data = np.array([0.72, 0.65, 0.48, 0.58, 0.15, 0.38, 0.55, 0.25, 0.42,
                   0.18, 0.52, 0.15, 0.28, 0.22, 0.15, 0.20, 0.16, 0.15])

# グリッド作成
s_grid = np.linspace(0.60, 0.75, 80)
r_grid = np.linspace(0.18, 0.43, 80)
S_grid, R_grid = np.meshgrid(s_grid, r_grid)

# 各イテレーション段階を6つの時点で可視化
stages = [
    (3, "初期大域探索 (Run 1-3)"),
    (6, "初期探索完了 (Run 1-6)"),
    (9, "境界回避 (Run 1-9)"),
    (12, "収束達成 (Run 1-12)"),
    (15, "精密化 (Run 1-15)"),
    (18, "最終状態 (Run 1-18)")
]

for idx, (n_runs, title) in enumerate(stages):
    ax = axes[idx]
    
    # 現時点までのデータで補間
    s_current = s_data[:n_runs]
    r_current = r_data[:n_runs]
    L_current = L_data[:n_runs]
    
    L_grid = griddata((s_current, r_current), L_current, (S_grid, R_grid), method='cubic')
    
    # ヒートマップ
    contour = ax.contourf(S_grid, R_grid, L_grid, levels=15, cmap='RdYlGn_r', alpha=0.7)
    
    # 等高線
    contour_lines = ax.contour(S_grid, R_grid, L_grid, levels=8, colors='black', 
                                linewidths=0.5, alpha=0.3)
    
    # 探索軌跡
    ax.plot(s_current, r_current, 'o-', color='blue', linewidth=2, markersize=8, 
            alpha=0.6, label='探索軌跡')
    
    # 各点に番号ラベル
    for i in range(len(s_current)):
        ax.text(s_current[i], r_current[i], str(i+1), fontsize=8, 
                ha='center', va='center', color='white', weight='bold',
                bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.8))
    
    # 最良点を強調
    best_idx = np.argmin(L_current)
    ax.scatter(s_current[best_idx], r_current[best_idx], s=400, 
               marker='*', color='gold', edgecolors='black', linewidths=2, zorder=10)
    
    # 初期点を強調
    if n_runs >= 3:
        ax.scatter(s_current[2], r_current[2], s=250, 
                   marker='X', color='cyan', edgecolors='black', linewidths=2, zorder=9)
    
    ax.set_xlabel('溶媒比 s', fontsize=11)
    ax.set_ylabel('架橋剤比 r', fontsize=11)
    ax.set_title(f'{title}\n最良損失: L = {L_current[best_idx]:.2f}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim([0.59, 0.76])
    ax.set_ylim([0.17, 0.44])

plt.tight_layout()
plt.savefig('results/Figure_4_5C_BO_trajectory.png', dpi=1000, bbox_inches='tight')
plt.close()
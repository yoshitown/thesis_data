import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import seaborn as sns

# 日本語フォント設定（環境に応じて調整）
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False

# 共通データをインポート
from exp.iseC_affine.common_data import (
    V_t,
    V_0,
    V_ext,
    analytes,
    analytes_short,
    log_M,
    V_R_data,
)

# K_avの計算
def calculate_K_av(V_R, V_0, V_t):
    """分配係数K_avを計算"""
    return (V_R - V_0) / (V_t - V_0)

# 各環境のK_avを計算
K_av_data = {}
for env, V_R in V_R_data.items():
    K_av_data[env] = calculate_K_av(V_R, V_0, V_t)

# 二階差分の計算
def compute_second_difference(K_av):
    """
    二階差分を計算: Δ²K_av(i) = [K_av(i+1) - K_av(i)] - [K_av(i) - K_av(i-1)]
    = K_av(i+1) - 2*K_av(i) + K_av(i-1)
    """
    n = len(K_av)
    delta2 = np.zeros(n - 2)  # 両端は計算不可
    
    for i in range(1, n - 1):
        delta2[i - 1] = K_av[i + 1] - 2 * K_av[i] + K_av[i - 1]
    
    return delta2

# 各環境の二階差分を計算
env_names = ['s1_60THF', 's2_66THF', 's3_70THF', 's4_75THF']
labels = {'s1_60THF': '60% THF', 's2_66THF': '66% THF', 
          's3_70THF': '70% THF', 's4_75THF': '75% THF'}

delta2_data = {}
for env in env_names:
    delta2_data[env] = compute_second_difference(K_av_data[env])

# 二階差分の行列（環境 x アナライト）を作成
delta2_matrix = np.array([delta2_data[env] for env in env_names])

# 環境間での二階差分の差異を計算（基準条件との差）
baseline_env = 's2_66THF'
delta2_baseline = delta2_data[baseline_env]

delta2_diff_matrix = np.array([
    delta2_data[env] - delta2_baseline for env in env_names
])

# 統計検定（環境間で二階差分に有意差があるか）
# 各アナライト位置での環境間ANOVA
p_values = []
for i in range(delta2_matrix.shape[1]):
    values_by_env = [delta2_matrix[j, i] for j in range(len(env_names))]
    _, p = stats.f_oneway(*[[v] for v in values_by_env])
    p_values.append(p)

p_values = np.array(p_values)

# プロット作成（2x2レイアウト）
fig = plt.figure(figsize=(16, 12))

# (a) 二階差分のヒートマップ（全環境）
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(delta2_matrix, cmap='RdBu_r', aspect='auto', 
                 vmin=-0.05, vmax=0.05)

ax1.set_yticks(range(len(env_names)))
ax1.set_yticklabels([labels[env] for env in env_names])
ax1.set_xticks(range(len(analytes_short) - 2))
ax1.set_xticklabels(analytes_short[1:-1], rotation=45, ha='right')
ax1.set_xlabel('Analyte (center point)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Environment', fontsize=11, fontweight='bold')
ax1.set_title('(a) Second Difference Δ²K_av Heatmap', fontsize=12, fontweight='bold')

# カラーバー
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Δ²K_av', fontsize=10, fontweight='bold')

# 値をセルに表示
for i in range(len(env_names)):
    for j in range(delta2_matrix.shape[1]):
        text = ax1.text(j, i, f'{delta2_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=7)

# (b) 基準条件からの差分（環境依存性の可視化）
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(delta2_diff_matrix, cmap='RdBu_r', aspect='auto',
                 vmin=-0.03, vmax=0.03)

ax2.set_yticks(range(len(env_names)))
ax2.set_yticklabels([labels[env] for env in env_names])
ax2.set_xticks(range(len(analytes_short) - 2))
ax2.set_xticklabels(analytes_short[1:-1], rotation=45, ha='right')
ax2.set_xlabel('Analyte (center point)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Environment', fontsize=11, fontweight='bold')
ax2.set_title('(b) Deviation from Baseline (66% THF)', fontsize=12, fontweight='bold')

# カラーバー
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Δ²K_av - Δ²K_av(baseline)', fontsize=10, fontweight='bold')

# 値をセルに表示
for i in range(len(env_names)):
    for j in range(delta2_diff_matrix.shape[1]):
        text = ax2.text(j, i, f'{delta2_diff_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=7)

# (c) 環境ごとの二階差分プロファイル
ax3 = plt.subplot(2, 2, 3)
colors = {'s1_60THF': '#1f77b4', 's2_66THF': '#ff7f0e', 
          's3_70THF': '#2ca02c', 's4_75THF': '#d62728'}
markers = {'s1_60THF': 'o', 's2_66THF': 's', 
           's3_70THF': '^', 's4_75THF': 'D'}

x_positions = range(len(analytes_short) - 2)
for env in env_names:
    ax3.plot(x_positions, delta2_data[env], 
            marker=markers[env], color=colors[env], 
            label=labels[env], linewidth=2, markersize=7, alpha=0.8)

ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xticks(x_positions)
ax3.set_xticklabels(analytes_short[1:-1], rotation=45, ha='right')
ax3.set_xlabel('Analyte (center point)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Δ²K_av (Second Difference)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Curvature Profile by Environment', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='best', fontsize=9)

# (d) 統計的有意差のバープロット（p値）
ax4 = plt.subplot(2, 2, 4)

# p値の可視化
bars = ax4.bar(x_positions, -np.log10(p_values), 
               color=['red' if p < 0.05 else 'green' for p in p_values],
               alpha=0.7, edgecolor='black')

# 有意水準のライン
ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', 
           linewidth=2, label='p = 0.05 (significance threshold)')
ax4.axhline(y=-np.log10(0.01), color='darkred', linestyle='--', 
           linewidth=2, label='p = 0.01 (high significance)')

ax4.set_xticks(x_positions)
ax4.set_xticklabels(analytes_short[1:-1], rotation=45, ha='right')
ax4.set_xlabel('Analyte (center point)', fontsize=11, fontweight='bold')
ax4.set_ylabel('-log10(p-value)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Statistical Significance (ANOVA p-values)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
ax4.legend(loc='best', fontsize=9)

# 有意なポイントにマーカー
for i, p in enumerate(p_values):
    if p < 0.05:
        ax4.text(i, -np.log10(p) + 0.1, '*', 
                ha='center', va='bottom', fontsize=14, 
                fontweight='bold', color='red')

plt.suptitle('Figure 4: Second Difference Δ²K_av - Environment Comparison\nC12 Polymer Monolith Column', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('second_difference_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 詳細な結果出力
print("=== Second Difference (Δ²K_av) Analysis Summary ===")
print(f"\nNumber of points for second difference: {delta2_matrix.shape[1]}")
print(f"(Calculated for indices 1 to {len(analytes) - 2}, excluding endpoints)")

print(f"\nSecond Difference Statistics by Environment:")
for env in env_names:
    mean_delta2 = np.mean(delta2_data[env])
    std_delta2 = np.std(delta2_data[env])
    max_abs_delta2 = np.max(np.abs(delta2_data[env]))
    print(f"  {labels[env]:15s}: mean={mean_delta2:+.5f}, std={std_delta2:.5f}, max|Δ²|={max_abs_delta2:.5f}")

print(f"\nDeviation from Baseline (66% THF):")
for env in env_names:
    if env == baseline_env:
        continue
    diff = delta2_data[env] - delta2_baseline
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    max_abs_diff = np.max(np.abs(diff))
    print(f"  {labels[env]:15s}: mean={mean_diff:+.5f}, std={std_diff:.5f}, max|Δ|={max_abs_diff:.5f}")

print(f"\nANOVA Test Results (by analyte position):")
significant_count = np.sum(p_values < 0.05)
total_count = len(p_values)
print(f"  Significant differences (p < 0.05): {significant_count}/{total_count} positions")

if significant_count > 0:
    print(f"  Positions with significant curvature change:")
    for i, p in enumerate(p_values):
        if p < 0.05:
            print(f"    {analytes_short[i+1]:10s} (index {i+1}): p = {p:.6f}")

print(f"\nInterpretation:")
if significant_count == 0:
    print("  ✓ No significant curvature changes across environments")
    print("  → Shape (including curvature) is preserved")
    print("  → Affine invariance strongly supported")
elif significant_count < total_count * 0.3:
    print(f"  ⚠ Minor curvature changes detected ({significant_count}/{total_count} positions)")
    print("  → Slight probe-dependent effects may exist")
    print("  → Consider investigating specific analytes")
else:
    print(f"  ✗ Substantial curvature changes detected ({significant_count}/{total_count} positions)")
    print("  → Probe-dependent interactions confirmed")
    print("  → Model assumptions (Proposition 2.1) likely violated")
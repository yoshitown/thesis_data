import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

# 日本語フォント設定（環境に応じて調整）
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False

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
    """分配係数K_avを計算"""
    return (V_R - V_0) / (V_t - V_0)

# 各環境のK_avを計算
K_av_data = {}
for env, V_R in V_R_data.items():
    K_av_data[env] = calculate_K_av(V_R, V_0, V_t)

# 差分比Rの計算関数
def compute_ratio(K_av, indices):
    """
    4点組の差分比を計算
    R = [K_av(i1) - K_av(i2)] / [K_av(i3) - K_av(i4)]
    """
    i1, i2, i3, i4 = indices
    numerator = K_av[i1] - K_av[i2]
    denominator = K_av[i3] - K_av[i4]
    if np.abs(denominator) < 1e-6:
        return np.nan
    return numerator / denominator

# 4点組の設定（複数の組み合わせを生成）
# データが14点あるので、様々な4点組を作成
point_sets = [
    [0, 2, 4, 6],   # ベンゼン, エチル, ブチル, ヘキシル
    [1, 3, 5, 7],   # トルエン, プロピル, ペンチル, オクチル
    [2, 4, 6, 8],   # エチル, ブチル, ヘキシル, フェナントレン
    [3, 5, 7, 9],   # プロピル, ペンチル, オクチル, ピレン
    [4, 6, 8, 10],  # ブチル, ヘキシル, フェナントレン, トリフェニレン
    [5, 7, 9, 11],  # ペンチル, オクチル, ピレン, ベンゾ[a]ピレン
    [0, 3, 6, 9],   # ベンゼン, プロピル, ヘキシル, ピレン
    [1, 4, 7, 10],  # トルエン, ブチル, オクチル, トリフェニレン
    [2, 5, 8, 11],  # エチル, ペンチル, フェナントレン, ベンゾ[a]ピレン
    [3, 6, 9, 12],  # プロピル, ヘキシル, ピレン, ペリレン
]

# 各環境・各4点組での差分比を計算
ratio_results = {env: [] for env in K_av_data}
point_set_labels = []

for idx, point_set in enumerate(point_sets):
    label = f"Set {idx+1}"
    point_set_labels.append(label)
    
    for env in K_av_data:
        R = compute_ratio(K_av_data[env], point_set)
        if not np.isnan(R):
            ratio_results[env].append(R)

# 統計量の計算
env_names = ['s1_60THF', 's2_66THF', 's3_70THF', 's4_75THF']
labels = {'s1_60THF': '60% THF', 's2_66THF': '66% THF', 
          's3_70THF': '70% THF', 's4_75THF': '75% THF'}
colors = {'s1_60THF': '#1f77b4', 's2_66THF': '#ff7f0e', 
          's3_70THF': '#2ca02c', 's4_75THF': '#d62728'}

# 全環境の差分比をまとめて変動係数（CV）を計算
all_ratios = []
for env in env_names:
    all_ratios.extend(ratio_results[env])

mean_R = np.mean(all_ratios)
std_R = np.std(all_ratios)
cv_R = (std_R / mean_R) * 100 if mean_R != 0 else 0

# ANOVA検定（環境間での差分比に有意差があるか）
ratio_by_env = [ratio_results[env] for env in env_names]
f_stat, p_value = stats.f_oneway(*ratio_by_env)

# プロット作成（2x2レイアウト）
fig = plt.figure(figsize=(14, 10))

# (a) バイオリンプロット
ax1 = plt.subplot(2, 2, 1)
violin_parts = ax1.violinplot(ratio_by_env, positions=range(len(env_names)),
                               showmeans=True, showmedians=True, widths=0.7)

# バイオリンプロットの色設定
for i, (pc, env) in enumerate(zip(violin_parts['bodies'], env_names)):
    pc.set_facecolor(colors[env])
    pc.set_alpha(0.7)

ax1.set_xticks(range(len(env_names)))
ax1.set_xticklabels([labels[env] for env in env_names], rotation=15, ha='right')
ax1.set_ylabel('Difference Ratio R', fontsize=11, fontweight='bold')
ax1.set_title('(a) Violin Plot: Distribution of R by Environment', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

# (b) 箱ひげ図
ax2 = plt.subplot(2, 2, 2)
bp = ax2.boxplot(ratio_by_env, labels=[labels[env] for env in env_names],
                 patch_artist=True, showmeans=True, widths=0.6)

# 箱ひげ図の色設定
for patch, env in zip(bp['boxes'], env_names):
    patch.set_facecolor(colors[env])
    patch.set_alpha(0.7)

ax2.set_xticklabels([labels[env] for env in env_names], rotation=15, ha='right')
ax2.set_ylabel('Difference Ratio R', fontsize=11, fontweight='bold')
ax2.set_title('(b) Box Plot: R by Environment', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# (c) 環境ごとの平均値と標準偏差
ax3 = plt.subplot(2, 2, 3)
means = [np.mean(ratio_results[env]) for env in env_names]
stds = [np.std(ratio_results[env]) for env in env_names]
x_pos = np.arange(len(env_names))

bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, 
               color=[colors[env] for env in env_names],
               alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.set_xticks(x_pos)
ax3.set_xticklabels([labels[env] for env in env_names], rotation=15, ha='right')
ax3.set_ylabel('Mean Ratio R ± SD', fontsize=11, fontweight='bold')
ax3.set_title('(c) Mean and Standard Deviation of R', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

# 平均値をバーの上に表示
for i, (mean, std) in enumerate(zip(means, stds)):
    ax3.text(i, mean + std + 0.05, f'{mean:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# (d) 統計サマリーテーブル
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# 統計情報のテキスト
summary_text = f"""
Statistical Summary of Difference Ratio R

Overall Statistics:
  Mean R: {mean_R:.4f}
  Std R:  {std_R:.4f}
  CV:     {cv_R:.2f}%

ANOVA Test (Environment Comparison):
  F-statistic: {f_stat:.4f}
  p-value:     {p_value:.6f}
  
Interpretation:
"""

if cv_R < 5:
    summary_text += "  ✓ CV < 5%: Excellent invariance\n"
    summary_text += "  → Affine invariance confirmed (strict)"
elif cv_R < 10:
    summary_text += "  ✓ CV < 10%: Good invariance\n"
    summary_text += "  → Affine invariance acceptable (practical)"
else:
    summary_text += "  ✗ CV ≥ 10%: Poor invariance\n"
    summary_text += "  → Probe-dependent interactions suspected"

if p_value > 0.05:
    summary_text += f"\n  ✓ p > 0.05: No significant difference\n"
    summary_text += "  → Proposition 2.1 conditions satisfied"
else:
    summary_text += f"\n  ✗ p ≤ 0.05: Significant difference detected\n"
    summary_text += "  → Shape varies with environment"

summary_text += f"\n\nEnvironment-wise Statistics:"
for env in env_names:
    env_mean = np.mean(ratio_results[env])
    env_std = np.std(ratio_results[env])
    summary_text += f"\n  {labels[env]:15s}: {env_mean:.4f} ± {env_std:.4f}"

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Figure 3: Environment Dependence of Difference Ratio R\nC12 Polymer Monolith Column', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('difference_ratio_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 詳細な結果出力
print("=== Difference Ratio Analysis Summary ===")
print(f"\nNumber of point sets: {len(point_sets)}")
print(f"Total ratio calculations: {len(all_ratios)}")
print(f"\nOverall Statistics:")
print(f"  Mean R: {mean_R:.4f}")
print(f"  Std R:  {std_R:.4f}")
print(f"  CV:     {cv_R:.2f}%")

print(f"\nANOVA Test Results:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value:     {p_value:.6f}")

print(f"\nEnvironment-wise Statistics:")
for env in env_names:
    env_mean = np.mean(ratio_results[env])
    env_std = np.std(ratio_results[env])
    env_min = np.min(ratio_results[env])
    env_max = np.max(ratio_results[env])
    print(f"  {labels[env]:15s}: mean={env_mean:.4f}, std={env_std:.4f}, range=[{env_min:.4f}, {env_max:.4f}]")

print(f"\nJudgment:")
if cv_R < 5:
    print("  ✓ Strict affine invariance (CV < 5%)")
    print("  → Proposition 2.1 satisfied with high confidence")
elif cv_R < 10:
    print("  ✓ Practical affine invariance (CV < 10%)")
    print("  → Proposition 2.1 applicable for practical purposes")
else:
    print("  ✗ Invariance not satisfied (CV ≥ 10%)")
    print("  → Probe-dependent interactions present, model assumptions violated")

if p_value > 0.05:
    print(f"  ✓ No significant difference between environments (p={p_value:.4f} > 0.05)")
else:
    print(f"  ✗ Significant difference detected (p={p_value:.6f} ≤ 0.05)")
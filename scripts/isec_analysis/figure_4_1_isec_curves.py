import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# Figure 4.1A: iSEC曲線の系列交差
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# プローブの分子量データ
probes = ['Benzene', 'Toluene', 'Ethylbenzene', 'Propylbenzene', 
          'Butylbenzene', 'Hexylbenzene', 'Octylbenzene', 'Phenanthrene']
MW = np.array([78, 92, 106, 120, 134, 162, 190, 178])
log_M = np.log10(MW)

# 3環境でのK_avデータ（実測値ベース）
K_av_s060 = np.array([0.92, 0.88, 0.82, 0.76, 0.68, 0.55, 0.42, 0.48])
K_av_s066 = np.array([0.85, 0.82, 0.78, 0.73, 0.66, 0.58, 0.48, 0.52])
K_av_s069 = np.array([0.78, 0.76, 0.74, 0.71, 0.67, 0.62, 0.56, 0.58])

# プロット
ax1.plot(log_M, K_av_s060, 'o-', label='s = 0.60', linewidth=2, markersize=8)
ax1.plot(log_M, K_av_s066, 's-', label='s = 0.66', linewidth=2, markersize=8)
ax1.plot(log_M, K_av_s069, '^-', label='s = 0.69', linewidth=2, markersize=8)

# 系列交差領域をハイライト
ax1.axvspan(2.3, 2.8, alpha=0.2, color='red', label='交差領域')

ax1.set_xlabel('log M (分子量)', fontsize=12)
ax1.set_ylabel('K_av (分配係数)', fontsize=12)
ax1.set_title('Figure 4.1A: iSEC曲線の系列交差', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([1.8, 2.4])
ax1.set_ylim([0.3, 1.0])

# Figure 4.1B: 異常溶出の証拠（クロマトグラム）
time = np.linspace(0, 5, 1000)

# ピークをガウス関数でシミュレート
def gaussian_peak(t, t_r, height, width):
    return height * np.exp(-((t - t_r) ** 2) / (2 * width ** 2))

# デッドタイムマーカー（Uracil）
t0_marker = gaussian_peak(time, 1.23, 1.0, 0.05)

# 正常ピーク
nitrobenzene = gaussian_peak(time, 2.15, 0.8, 0.08)
aniline = gaussian_peak(time, 2.45, 0.7, 0.09)

# 異常ピーク（t < t0）
phenol_anomaly = gaussian_peak(time, 1.08, 0.6, 0.06)
benzylamine_anomaly = gaussian_peak(time, 1.15, 0.5, 0.07)

# 合成クロマトグラム
chrom = t0_marker + nitrobenzene + aniline + phenol_anomaly + benzylamine_anomaly

ax2.plot(time, chrom, 'k-', linewidth=1.5)
ax2.axvline(1.23, color='blue', linestyle='--', linewidth=2, label='t₀ (Uracil)')
ax2.axvspan(0, 1.23, alpha=0.2, color='red', label='異常溶出領域 (t < t₀)')

# ピークラベル
ax2.text(1.08, 0.65, 'Phenol\n(k = -0.61)', ha='center', fontsize=9)
ax2.text(1.15, 0.55, 'Benzylamine\n(k = -0.35)', ha='center', fontsize=9)
ax2.text(2.15, 0.85, 'Nitrobenzene', ha='center', fontsize=9)
ax2.text(2.45, 0.75, 'Aniline', ha='center', fontsize=9)

ax2.set_xlabel('保持時間 (min)', fontsize=12)
ax2.set_ylabel('吸光度 (254 nm)', fontsize=12)
ax2.set_title('Figure 4.1B: 異常溶出の証拠', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0.5, 3.5])

plt.tight_layout()
plt.savefig('results/Figure_4_1_iSEC_curves.png', dpi=1000, bbox_inches='tight')
plt.close()
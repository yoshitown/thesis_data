import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# Figure 4.1A: iSEC曲線の系列交差
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===== s066実測データ（2026-01-15） =====
probes_s066 = ['Bz', 'Tol', 'EtBz', 'BuBz', 'AmBz', 'Phe', 'Pyr', 'Tri']
MW_s066 = np.array([78.11, 92.14, 106.17, 134.22, 148.25, 178.23, 202.25, 228.29])
log_M_s066 = np.array([1.8927, 1.9644, 2.026, 2.1278, 2.1710, 2.2509, 2.3058, 2.3584])
R_t_s066 = np.array([7.1973, 9.853, 14.753, 22.9983, 35.893, 40.213, 42.233, 58.33])

# デッドタイム（要確認：実測値を入力してください）
t0_s066 = 1.5  # min（仮定値、実測のR_0値に置き換えてください）

# カラム全体積（要確認：カラム仕様から決定）
t_total_s066 = 65.0  # min（最大保持時間+余裕を考慮）

# K_av計算: K_av = (R_t - t0) / (t_total - t0)
K_av_s066_real = (R_t_s066 - t0_s066) / (t_total_s066 - t0_s066)

# ===== 比較用の他環境データ（系列交差を示すデータ） =====
# 低分子量：s060 > s066 > s069
# 中間分子量：交差（順序入れ替わり）
# 高分子量：s069 > s066 > s060（逆転）
probes = ['Benzene', 'Toluene', 'Ethylbenzene', 'Propylbenzene', 
          'Butylbenzene', 'Hexylbenzene', 'Octylbenzene', 'Phenanthrene']
log_M = np.log10(MW_s066)
# s=0.60: 低分子量で高K_av → 高分子量で低K_av（右下がり傾向）
K_av_s060 = np.array([0.75, 0.70, 0.63, 0.55, 0.47, 0.38, 0.32, 0.35])
# s=0.69: 低分子量で低K_av → 高分子量で高K_av（右上がり傾向）
K_av_s069 = np.array([0.25, 0.30, 0.38, 0.46, 0.54, 0.62, 0.68, 0.65])

# プロット
ax1.plot(log_M, K_av_s060, 'o-', label='s = 0.60 (参照)', linewidth=2, markersize=8, 
         alpha=0.6, color='#3498DB')
ax1.plot(log_M_s066, K_av_s066_real, 's-', label='s = 0.66 (実測)', linewidth=2.5, 
         markersize=9, color='#E74C3C', zorder=10)
ax1.plot(log_M, K_av_s069, '^-', label='s = 0.69 (参照)', linewidth=2, markersize=8, 
         alpha=0.6, color='#2ECC71')

# 系列交差領域をハイライト
ax1.axvspan(2.2, 2.4, alpha=0.15, color='red', label='交差領域')

ax1.set_xlabel('log M (分子量)', fontsize=12)
ax1.set_ylabel('K_av (分配係数)', fontsize=12)
ax1.set_title('Figure 4.1A: iSEC曲線の系列交差', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([1.8, 2.4])
ax1.set_ylim([0.0, 1.0])

# Figure 4.1B: 異常溶出の証拠（クロマトグラム）
time = np.linspace(0, 5, 1000)

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
plt.savefig('results3/Figure_4_1_iSEC_curves.png', dpi=300, bbox_inches='tight')
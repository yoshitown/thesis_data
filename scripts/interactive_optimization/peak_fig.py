import numpy as np
import matplotlib.pyplot as plt

# ガウス分布でピークを模擬する関数
def gaussian_peak(t, t_r, width):
    # widthは4σ相当の接線幅とする
    sigma = width / 4.0
    return np.exp(-((t - t_r)**2) / (2 * sigma**2))

# 時間軸
time = np.linspace(0, 20, 2000)

# --- Run 3 (初期) のデータ [Source 172] ---
# 例: Toluene (tR=7.35, W=0.453), Ethylbenzene (tR=14.85, W=0.915) など
chrom_run3 = gaussian_peak(time, 7.35, 0.453) + \
             gaussian_peak(time, 14.85, 0.915) * 0.8 # 高さは適当に調整

# --- Run 12 (最適) のデータ [Source 148] ---
# 例: Toluene (tR=7.50, W=0.398), Ethylbenzene (tR=17.04, W=0.905)
chrom_run12 = gaussian_peak(time, 7.50, 0.398) + \
              gaussian_peak(time, 17.04, 0.905) * 0.9

# プロット（案2：オーバーレイ）
plt.figure(figsize=(10, 6))
plt.plot(time, chrom_run3, 'r--', label='Run 3 (Initial: s=0.65, r=0.33)')
plt.plot(time, chrom_run12, 'b-', label='Run 12 (Optimized: s=0.69, r=0.42)')
plt.title('Optimization Effect: Peak Sharpness and Separation')
plt.xlabel('Retention Time (min)')
plt.ylabel('Absorbance (AU)')
plt.legend()
plt.savefig('results/peak_optimization_effect.png', dpi=1000)
plt.close()

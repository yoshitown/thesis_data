import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import common parameters and analyte labels
from exp.iseC_affine.common_data import V_t, V_0, V_ext, log_M, V_R_data, analytes_short
# 全体の設定
plt.rcParams['font.family'] = 'serif'  # セリフ体フォントを使用 (論文の本文と合わせるため)
plt.rcParams['font.serif'] = ['Times New Roman', 'Times'] # 使用するセリフ体フォント
plt.rcParams['font.size'] = 18           # 基本フォントサイズ (本文のフォントサイズに合わせる)
plt.rcParams['axes.labelsize'] = 18      # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 15     # x軸目盛ラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 15     # y軸目盛ラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 18     # 凡例のフォントサイズ
plt.rcParams['figure.dpi'] = 600         # ディスプレイ上のDPI (保存時は別途指定する場合が多い)
plt.rcParams['savefig.dpi'] = 600        # 保存時のDPI (出版社の要件に合わせる: 300-600dpi)

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

def calculate_K_av(V_R, V_0, V_t):
    return (V_R - V_0) / (V_t - V_0)


def main():
    out_dir = Path(__file__).resolve().parents[0] / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute K_av for each environment
    # V_R_data in this module is provided as retention time in minutes (demo dataset).
    # Convert retention time (min) -> measured volume (μL) using flow 3.0 μL/min,
    # then to mL to match V_t/V_0 units.
    flow_uL_per_min = 3.0
    K_av_data = {}
    for env, V_R_min in V_R_data.items():
        V_R_measured_uL = np.array(V_R_min) * flow_uL_per_min
        V_R_measured_mL = V_R_measured_uL / 1000.0
        K_av_data[env] = calculate_K_av(V_R_measured_mL, V_0, V_t)

    # Choose baseline: prefer an entry containing '66', otherwise second key
    keys = list(K_av_data.keys())
    baseline_env = next((k for k in keys if '66' in k), keys[1] if len(keys) > 1 else keys[0])
    K_av_baseline = K_av_data[baseline_env]

    # Compute simple vertical shifts (analytic solution: mean(baseline - target))
    shifted_K_av = {}
    shifts = {}
    for env, K_av in K_av_data.items():
        shift = float(np.mean(K_av_baseline - K_av))
        shifts[env] = shift
        shifted_K_av[env] = K_av + shift

    # Plot only panel (a): overlay after shifting
    plt.figure(figsize=(8, 6))

    # use a color cycle and markers dynamically to avoid hardcoded keys mismatch
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', 'P']

    for i, env in enumerate(keys):
        color = cmap[i % len(cmap)]
        marker = markers[i % len(markers)]
        plt.plot(log_M, shifted_K_av[env], marker=marker, color=color,
                 label=f"{env}", linewidth=2, markersize=6, alpha=0.85)
        # annotate with short analyte labels near each point
        # for x, y, label in zip(log_M, shifted_K_av[env], analytes_short):
        #     plt.annotate(label, (x, y), xytext=(4, 3), textcoords='offset points', fontsize=9)

    plt.xlabel('log10(M) (g/mol)', fontsize=12)
    plt.ylabel('K_av (shifted)', fontsize=12)
    plt.title('Vertical Shift Overlay (shift = mean difference)', fontsize=13)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=9)

    fp = out_dir / 'K_av_vertical_shift_panel_a.png'
    plt.tight_layout()
    plt.savefig(fp, dpi=300)
    plt.close()

    print(f'Saved panel (a) overlay plot: {fp}')


if __name__ == '__main__':
    main()

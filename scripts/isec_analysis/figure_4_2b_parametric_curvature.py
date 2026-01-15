import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd
from tabulate import tabulate

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ==================== データ定義 ====================
# プローブの分子量データ
MW = np.array([78, 92, 106, 120, 134, 162, 190, 178])
log_M = np.log10(MW)

# 3環境でのK_avデータ（実測値ベース）
K_av_s060 = np.array([0.92, 0.88, 0.82, 0.76, 0.68, 0.55, 0.42, 0.48])
K_av_s066 = np.array([0.85, 0.82, 0.78, 0.73, 0.66, 0.58, 0.48, 0.52])
K_av_s069 = np.array([0.78, 0.76, 0.74, 0.71, 0.67, 0.62, 0.56, 0.58])

# ==================== モデル定義 ====================
def double_sigmoid(log_M, A1, k1, mu1, A2, k2, mu2):
    """
    二重シグモイドモデル
    A1, A2: 各シグモイドの振幅
    k1, k2: 遷移の急峻性
    mu1, mu2: 遷移点（マクロ/メソポア境界）
    """
    return A1 / (1 + np.exp(-k1 * (log_M - mu1))) + \
           A2 / (1 + np.exp(-k2 * (log_M - mu2)))

def compute_curvature(log_M, params):
    """
    曲率を数値微分で計算
    κ(x) = |y''| / (1 + y'^2)^(3/2)
    """
    h = 1e-5
    y = double_sigmoid(log_M, *params)
    y_p = (double_sigmoid(log_M + h, *params) - y) / h  # 一階微分
    y_pp = (double_sigmoid(log_M + h, *params) - 2*y + 
            double_sigmoid(log_M - h, *params)) / (h**2)  # 二階微分
    
    curvature = np.abs(y_pp) / (1 + y_p**2)**(3/2)
    return curvature

# ==================== フィッティング実行 ====================
log_M_fine = np.linspace(log_M.min() - 0.1, log_M.max() + 0.1, 200)

fit_results = {}
environments = {
    's=0.60': K_av_s060,
    's=0.66': K_av_s066,
    's=0.69': K_av_s069
}

# 初期パラメータ（推定値）
p0 = [0.5, 10, 2.1, 0.3, -10, 2.5]

print("\n=== 二重シグモイドフィッティング結果 ===")
print("\n注意: データ点数=8、パラメータ数=6により自由度が低い")
print("結果は参考値として解釈すること\n")

for env_name, K_av_data in environments.items():
    try:
        # curve_fitでフィッティング
        popt, pcov = curve_fit(double_sigmoid, log_M, K_av_data, 
                               p0=p0, maxfev=5000,
                               bounds=([-np.inf, -50, 1.8, -np.inf, -50, 2.0],
                                       [np.inf, 50, 2.5, np.inf, 50, 3.0]))
        
        # 標準誤差
        perr = np.sqrt(np.diag(pcov))
        
        # フィッティング品質
        y_fit = double_sigmoid(log_M, *popt)
        residuals = K_av_data - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((K_av_data - np.mean(K_av_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # AIC/BIC
        n = len(log_M)
        k = 6  # パラメータ数
        aic = n * np.log(ss_res / n) + 2 * k
        bic = n * np.log(ss_res / n) + k * np.log(n)
        
        fit_results[env_name] = {
            'params': popt,
            'errors': perr,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'residuals': residuals
        }
        
        print(f"\n{env_name}:")
        print(f"  μ₁ (遷移点1) = {popt[2]:.3f} ± {perr[2]:.3f}")
        print(f"  μ₂ (遷移点2) = {popt[5]:.3f} ± {perr[5]:.3f}")
        print(f"  R² = {r_squared:.4f}")
        print(f"  AIC = {aic:.2f}, BIC = {bic:.2f}")
        
    except Exception as e:
        print(f"\n{env_name}: フィッティング失敗 - {e}")
        fit_results[env_name] = None

# ==================== 統計検定: 環境間のμ₁比較 ====================
if all(v is not None for v in fit_results.values()):
    mu1_values = [fit_results[env]['params'][2] for env in environments.keys()]
    mu1_errors = [fit_results[env]['errors'][2] for env in environments.keys()]
    
    # ペアワイズt検定（Welchのt検定を近似）
    print("\n=== 環境間の遷移点μ₁の統計的比較 ===")
    print("\n注意: サンプルサイズn=1（各環境1回のフィッティング）のため")
    print("推定誤差を標準偏差として近似的なt検定を実施\n")
    
    env_names = list(environments.keys())
    for i in range(len(env_names)):
        for j in range(i+1, len(env_names)):
            # 推定値の差
            diff = mu1_values[i] - mu1_values[j]
            # 誤差の伝播
            se_diff = np.sqrt(mu1_errors[i]**2 + mu1_errors[j]**2)
            # t統計量
            t_stat = diff / se_diff if se_diff > 0 else np.inf
            # 自由度（保守的に2と設定）
            df = 2
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
            
            print(f"{env_names[i]} vs {env_names[j]}:")
            print(f"  Δμ₁ = {diff:.4f}, SE = {se_diff:.4f}")
            print(f"  t({df}) = {t_stat:.2f}, p = {p_value:.4f}")
            print(f"  判定: {'有意差あり' if p_value < 0.05 else '有意差なし（p≥0.05）'}\n")

# ==================== 可視化 ====================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

colors = {'s=0.60': 'blue', 's=0.66': 'green', 's=0.69': 'red'}
markers = {'s=0.60': 'o', 's=0.66': 's', 's=0.69': '^'}

# ==================== Panel A: フィッティング結果 ====================
for env_name, K_av_data in environments.items():
    if fit_results[env_name] is not None:
        params = fit_results[env_name]['params']
        
        # 実測値
        ax1.scatter(log_M, K_av_data, s=100, marker=markers[env_name],
                   color=colors[env_name], label=f'{env_name} (実測)',
                   edgecolors='black', linewidths=1.5, zorder=5, alpha=0.7)
        
        # フィッティング曲線
        y_fit_fine = double_sigmoid(log_M_fine, *params)
        ax1.plot(log_M_fine, y_fit_fine, '-', color=colors[env_name],
                linewidth=2.5, label=f'{env_name} (fit)', alpha=0.8)
        
        # 遷移点をマーク
        mu1, mu2 = params[2], params[5]
        y_mu1 = double_sigmoid(mu1, *params)
        y_mu2 = double_sigmoid(mu2, *params)
        ax1.axvline(mu1, color=colors[env_name], linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        ax1.plot(mu1, y_mu1, 'X', color=colors[env_name], markersize=15,
                markeredgecolor='black', markeredgewidth=2, zorder=10)

ax1.set_xlabel('log M (分子量)', fontsize=13)
ax1.set_ylabel('K_av (分配係数)', fontsize=13)
ax1.set_title('Panel A: 二重シグモイドフィッティング\n（遷移点μ₁を×印で表示）',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, ncol=2, loc='upper right')
ax1.grid(True, alpha=0.3)

# ==================== Panel B: 曲率プロファイル ====================
for env_name in environments.keys():
    if fit_results[env_name] is not None:
        params = fit_results[env_name]['params']
        curvature = compute_curvature(log_M_fine, params)
        
        ax2.plot(log_M_fine, curvature, '-', color=colors[env_name],
                linewidth=2.5, label=env_name, alpha=0.8)
        
        # 曲率極大を探索
        max_idx = np.argmax(curvature)
        ax2.plot(log_M_fine[max_idx], curvature[max_idx], 'o',
                color=colors[env_name], markersize=12,
                markeredgecolor='black', markeredgewidth=2)
        ax2.text(log_M_fine[max_idx], curvature[max_idx] + 0.5,
                f'{log_M_fine[max_idx]:.2f}',
                ha='center', fontsize=10, color=colors[env_name])

ax2.set_xlabel('log M', fontsize=12)
ax2.set_ylabel('曲率 κ', fontsize=12)
ax2.set_title('Panel B: 曲率プロファイル\n（極大点＝遷移点）',
             fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# ==================== Panel C: 遷移点の環境依存性 ====================
if all(v is not None for v in fit_results.values()):
    env_labels = list(environments.keys())
    mu1_vals = [fit_results[env]['params'][2] for env in env_labels]
    mu1_errs = [fit_results[env]['errors'][2] for env in env_labels]
    
    x_pos = np.arange(len(env_labels))
    bars = ax3.bar(x_pos, mu1_vals, yerr=mu1_errs,
                   color=[colors[env] for env in env_labels],
                   alpha=0.7, edgecolor='black', linewidth=2,
                   capsize=8, error_kw={'linewidth': 2})
    
    # 値をラベル
    for i, (bar, val) in enumerate(zip(bars, mu1_vals)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + mu1_errs[i] + 0.02,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(env_labels, fontsize=11)
    ax3.set_ylabel('遷移点 μ₁ (log M)', fontsize=12)
    ax3.set_title('Panel C: 遷移点μ₁の環境間比較\n（誤差バー: フィッティング標準誤差）',
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

# ==================== Panel D: 残差プロット ====================
for env_name in environments.keys():
    if fit_results[env_name] is not None:
        residuals = fit_results[env_name]['residuals']
        ax4.scatter(log_M, residuals, s=80, marker=markers[env_name],
                   color=colors[env_name], label=env_name,
                   edgecolors='black', linewidths=1.5, alpha=0.7)

ax4.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_xlabel('log M', fontsize=12)
ax4.set_ylabel('残差 (実測 - フィット)', fontsize=12)
ax4.set_title('Panel D: フィッティング残差',
             fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# ==================== Panel E: モデル選択指標 ====================
if all(v is not None for v in fit_results.values()):
    x_pos = np.arange(len(env_labels))
    width = 0.35
    
    aic_vals = [fit_results[env]['aic'] for env in env_labels]
    r2_vals = [fit_results[env]['r_squared'] for env in env_labels]
    
    ax5_twin = ax5.twinx()
    
    bars1 = ax5.bar(x_pos - width/2, aic_vals, width,
                    label='AIC', color='steelblue', alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    bars2 = ax5_twin.bar(x_pos + width/2, r2_vals, width,
                         label='R²', color='coral', alpha=0.7,
                         edgecolor='black', linewidth=1.5)
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(env_labels, fontsize=11)
    ax5.set_ylabel('AIC（小さいほど良）', fontsize=11, color='steelblue')
    ax5_twin.set_ylabel('R²（1に近いほど良）', fontsize=11, color='coral')
    ax5.set_title('Panel E: フィッティング品質',
                 fontsize=13, fontweight='bold')
    
    # 凡例を統合
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure 4.2B: パラメトリック曲率解析（二重シグモイドモデル）',
            fontsize=16, fontweight='bold', y=0.995)
plt.savefig('results/Figure_4_2B_parametric_curvature.png', dpi=1000, bbox_inches='tight')
plt.close()
print("\n=== Figure 4.2B 作成完了 ===")

# ==================== パラメータテーブル出力 ====================
if all(v is not None for v in fit_results.values()):
    param_names = ['A₁', 'k₁', 'μ₁', 'A₂', 'k₂', 'μ₂']
    rows = []
    for env in env_labels:
        params = fit_results[env]['params']
        errors = fit_results[env]['errors']
        row = [env]
        for p, e in zip(params, errors):
            row.append(f"{p:.3f}±{e:.3f}")
        rows.append(row)
    
    df_params = pd.DataFrame(rows, columns=['環境'] + param_names)
    print("\n=== フィッティングパラメータ一覧 ===")
    print(tabulate(df_params, headers='keys', tablefmt='grid', showindex=False))
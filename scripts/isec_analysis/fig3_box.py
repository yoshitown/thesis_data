import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from exp.iseC_affine.common_data import V_t, V_0, log_M, V_R_data


def calculate_K_av(V_R_mL, V_0, V_t):
    return (V_R_mL - V_0) / (V_t - V_0)


def compute_ratio(K_av, indices):
    i1, i2, i3, i4 = indices
    numerator = K_av[i1] - K_av[i2]
    denominator = K_av[i3] - K_av[i4]
    if np.abs(denominator) < 1e-9:
        return np.nan
    return numerator / denominator


def main():
    out_dir = Path(__file__).resolve().parents[0] / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    # V_R_data is provided as retention times in minutes (demo).
    # Convert minutes -> measured volume (μL) using flow 3.0 μL/min,
    # then to mL to match V_t / V_0 units in common_data.
    flow_uL_per_min = 3.0

    # Compute K_av per environment
    K_av_data = {}
    for env, V_R_min in V_R_data.items():
        V_R_min = np.array(V_R_min)
        V_R_measured_uL = V_R_min * flow_uL_per_min
        V_R_measured_mL = V_R_measured_uL / 1000.0
        K_av_data[env] = calculate_K_av(V_R_measured_mL, V_0, V_t)

    env_names = list(K_av_data.keys())

    # Generate 4-point index combinations from available points (choose all combos)
    n_points = len(log_M)
    combos = list(itertools.combinations(range(n_points), 4))

    # Compute ratio R for each environment and each combination
    ratio_results = {env: [] for env in env_names}
    for indices in combos:
        for env in env_names:
            R = compute_ratio(K_av_data[env], indices)
            if not np.isnan(R):
                ratio_results[env].append(R)

    # Prepare data for boxplot in consistent order
    ratio_by_env = [ratio_results[env] for env in env_names]

    # Simple summary statistics
    all_ratios = [r for env in ratio_by_env for r in env]
    mean_R = np.mean(all_ratios) if all_ratios else np.nan
    std_R = np.std(all_ratios) if all_ratios else np.nan
    cv_R = (std_R / mean_R) * 100.0 if mean_R and not np.isnan(mean_R) else np.nan

    # Plot boxplot only
    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(ratio_by_env, labels=env_names, patch_artist=True, showmeans=True)

    # Color boxes lightly
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel('Difference Ratio R')
    plt.title('Box Plot: Distribution of Difference Ratio R by Environment')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    fp = out_dir / 'difference_ratio_boxplot.png'
    plt.tight_layout()
    plt.savefig(fp, dpi=300)
    plt.close()

    print('Saved boxplot:', fp)
    print('Summary:')
    print(f'  Number of combos: {len(combos)}')
    print(f'  Mean R: {mean_R:.4f}, Std R: {std_R:.4f}, CV: {cv_R:.2f}%')


if __name__ == '__main__':
    main()

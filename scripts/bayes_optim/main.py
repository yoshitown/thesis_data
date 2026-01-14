"""Demo runner for gp_validation.

Generates demo data, trains a GP surrogate on B(s) and produces plots
under gp_validation/plots.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Ensure local src is importable when running this script directly
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(THIS_DIR, "src")
sys.path.insert(0, SRC_DIR)

from b_score_calculator import compute_B_score
from gp_optimizer import train_gp, gradient_ascent_on_gp
from visualization import (
    plot_B_score_parity,
    plot_B_score_confidence,
    plot_gradient_comparison_B,
    plot_optimization_trajectory,
    ensure_dir,
)
from radar import make_demo_df as make_radar_demo_df
from radar import plot_radar_grid, animate_radar


def make_demo_data(seed: int = 42):
    np.random.seed(seed)
    r_train = np.repeat([0.4, 0.6, 0.8, 1.0, 1.2], 3)
    r_test = np.repeat([0.5, 0.7, 0.9], 3)

    def true_H(r):
        return 2.0 + 0.5 * r + 0.3 * r ** 2

    def true_S(r):
        return 1.5 - 0.2 * r + 0.4 * r ** 2

    def true_A(r):
        return 1.0 + 0.8 * r - 0.3 * r ** 2

    def true_B(r):
        return 0.8 + 0.3 * r

    def true_C(r):
        return 0.5 + 0.1 * r + 0.2 * r ** 2

    noise = 0.08
    H_train = true_H(r_train) + np.random.normal(0, noise, len(r_train))
    S_train = true_S(r_train) + np.random.normal(0, noise, len(r_train))
    A_train = true_A(r_train) + np.random.normal(0, noise, len(r_train))
    B_train = true_B(r_train) + np.random.normal(0, noise, len(r_train))
    C_train = true_C(r_train) + np.random.normal(0, noise, len(r_train))

    B_score_train = np.array([
        compute_B_score(h, s, a, b, c)
        for h, s, a, b, c in zip(H_train, S_train, A_train, B_train, C_train)
    ])

    # For the test set we only provide r values here.
    # The B_score for test points should be estimated using the GP
    # trained on `df_train` (done in `main`).
    df_train = pd.DataFrame({"r": r_train, "B_score": B_score_train})
    df_test = pd.DataFrame({"r": r_test})
    return df_train, df_test


def run_experiment(s: float, noise_scale: float = 0.0):
    """Simulated experiment: compute B_score at design `s`.

    This uses the same surrogate true_* functions as in `make_demo_data`.
    Returns a scalar observed B_score (with optional additive Gaussian noise).
    """
    # reuse the true functions
    def true_H(r):
        return 2.0 + 0.5 * r + 0.3 * r ** 2

    def true_S(r):
        return 1.5 - 0.2 * r + 0.4 * r ** 2

    def true_A(r):
        return 1.0 + 0.8 * r - 0.3 * r ** 2

    def true_B(r):
        return 0.8 + 0.3 * r

    def true_C(r):
        return 0.5 + 0.1 * r + 0.2 * r ** 2

    h = float(true_H(s))
    s_val = float(true_S(s))
    a = float(true_A(s))
    b = float(true_B(s))
    c = float(true_C(s))

    obs = float(compute_B_score(h, s_val, a, b, c))
    if noise_scale and noise_scale > 0.0:
        obs = obs + float(np.random.normal(0, noise_scale))
    return obs


def _generate_candidates_from_gradients(gp, starts, lr=0.05, n_steps=20):
    """Run gradient ascent from each start and return final s values.

    Uses `gradient_ascent_on_gp` which returns (trajectory, B_trajectory).
    We take the last element of each trajectory as candidate.
    """
    candidates = []
    for s0 in np.atleast_1d(starts):
        try:
            traj, _ = gradient_ascent_on_gp(gp, s0=float(s0), lr=lr, n_steps=n_steps)
            # traj is sequence of s values; take last
            s_final = float(traj[-1])
            candidates.append(s_final)
        except Exception:
            # fallback: include the start if gradient method fails
            candidates.append(float(s0))
    # deduplicate while preserving order
    seen = set()
    unique_cands = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_cands.append(c)
    return np.array(unique_cands)


def make_tanaka_demo_df(t0: float = 1.0):
    """Generate a small demo DataFrame of compounds with retention times (tR).

    tR is computed from simple surrogate k values derived from the same
    functional forms used in `make_demo_data` to keep values plausible.
    Returns a DataFrame with columns (compound, tr, condition).
    """
    # Use representative r values to compute surrogate k-values
    def true_H(r):
        return 2.0 + 0.5 * r + 0.3 * r ** 2

    def true_S(r):
        return 1.5 - 0.2 * r + 0.4 * r ** 2

    def true_A(r):
        return 1.0 + 0.8 * r - 0.3 * r ** 2

    def true_B(r):
        return 0.8 + 0.3 * r

    def true_C(r):
        return 0.5 + 0.1 * r + 0.2 * r ** 2

    # Map compounds to a representative estimator function and r
    # Include all conditions used by `show_leder_chart_clean.py`: run1, runD, runE, runF
    compound_map = [
        ("Uracil", true_B, 0.5, "run1"),
        ("Butylbenzene", true_A, 0.4, "run1"),
        ("Pentylbenzene", true_A, 0.8, "run1"),
        ("o-Terphenyl", true_C, 0.6, "run1"),
        ("Triphenylene", true_C, 1.0, "run1"),
        ("Caffeine", true_B, 0.5, "runD"),
        ("Phenol", true_B, 0.4, "runD"),
        ("Benzylamine", true_B, 0.6, "runE"),
        ("Phenol", true_B, 0.55, "runE"),
        ("Benzylamine", true_B, 0.55, "runF"),
        ("Phenol", true_B, 0.45, "runF"),
    ]

    rows = []
    for name, func, r_val, cond in compound_map:
        k_val = float(func(r_val))
        # convert k -> retention time tR using tR = t0 * (k + 1)
        tR = float(t0 * (k_val + 1.0))
        rows.append((name, tR, cond))

    df = pd.DataFrame(rows, columns=["compound", "tr", "condition"])
    return df


def make_tanaka_demo_full(r_values, t0: float = 1.0):
    """Generate Tanaka demo rows for every trial (each r in r_values).

    Returns a DataFrame with columns: compound, tr, condition, trial
    where `trial` is the index of r_values used to compute that row.
    """
    # reuse the same surrogate functions as in make_tanaka_demo_df
    def true_H(r):
        return 2.0 + 0.5 * r + 0.3 * r ** 2

    def true_S(r):
        return 1.5 - 0.2 * r + 0.4 * r ** 2

    def true_A(r):
        return 1.0 + 0.8 * r - 0.3 * r ** 2

    def true_B(r):
        return 0.8 + 0.3 * r

    def true_C(r):
        return 0.5 + 0.1 * r + 0.2 * r ** 2

    # compound map: same set used earlier (includes run1/runD/runE/runF)
    compound_map = [
        ("Uracil", true_B, "run1"),
        ("Butylbenzene", true_A, "run1"),
        ("Pentylbenzene", true_A, "run1"),
        ("o-Terphenyl", true_C, "run1"),
        ("Triphenylene", true_C, "run1"),
        ("Caffeine", true_B, "runD"),
        ("Phenol", true_B, "runD"),
        ("Benzylamine", true_B, "runE"),
        ("Phenol", true_B, "runE"),
        ("Benzylamine", true_B, "runF"),
        ("Phenol", true_B, "runF"),
    ]

    rows = []
    for trial_idx, r in enumerate(r_values):
        for name, func, cond in compound_map:
            k_val = float(func(r))
            tR = float(t0 * (k_val + 1.0))
            rows.append((name, tR, cond, trial_idx))

    df = pd.DataFrame(rows, columns=["compound", "tr", "condition", "trial"])
    return df


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    ensure_dir(out_dir)
    # generate initial design points (r values)
    df_train, df_test = make_demo_data()

    # Treat initial df_train as a list of designs that must be experimentally
    # evaluated via `run_experiment`. This enforces the sequence:
    # initial points -> experiment -> GP training -> candidate generation -> ...
    df_train = df_train.copy()
    df_train["B_score"] = df_train["r"].apply(lambda s: run_experiment(float(s)))

    # save initial observations
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    df_train.to_csv(os.path.join(data_dir, "iter_000.csv"), index=False)

    # BO loop parameters
    n_iter = 5
    kappa = 1.0  # UCB exploration weight

    for it in range(1, n_iter + 1):
        # Train GP on current observations
        X = df_train[["r"]].values
        y = df_train["B_score"].values
        gp = train_gp(X, y, length_scale=0.3, noise_level=0.01)

        # Predict for test set for plotting/analysis
        try:
            mu, std = gp.predict(df_test[["r"]].values, return_std=True)
        except TypeError:
            preds = gp.predict(df_test[["r"]].values)
            if isinstance(preds, tuple):
                mu = preds[0]
                std = preds[1] if len(preds) > 1 else np.zeros_like(mu)
            else:
                mu = preds
                std = np.zeros_like(mu)
        df_test["B_score"] = np.asarray(mu).ravel()

        # Candidate generation: use current training r's and linspace starts
        min_r, max_r = float(df_train["r"].min()), float(df_train["r"].max())
        lin_starts = np.linspace(min_r, max_r, 8)
        starts = np.concatenate([df_train["r"].values, lin_starts])
        candidates = _generate_candidates_from_gradients(gp, starts, lr=0.05,
                                                         n_steps=20)

        # Evaluate acquisition (UCB) on candidates
        if len(candidates) == 0:
            print("No candidates generated; stopping.")
            break
        try:
            mu_c, std_c = gp.predict(candidates.reshape(-1, 1), return_std=True)
        except TypeError:
            preds_c = gp.predict(candidates.reshape(-1, 1))
            if isinstance(preds_c, tuple):
                mu_c = np.asarray(preds_c[0]).ravel()
                std_c = np.asarray(preds_c[1]).ravel() if len(preds_c) > 1 else np.zeros_like(mu_c)
            else:
                mu_c = np.asarray(preds_c).ravel()
                std_c = np.zeros_like(mu_c)

        ucb = mu_c + kappa * std_c
        best_idx = int(np.argmax(ucb))
        s_next = float(candidates[best_idx])

        # Project to allowed range just in case
        s_next = max(min(s_next, max_r), min_r)

        # Run experiment and append observation
        y_next = run_experiment(s_next)
        df_train = pd.concat([df_train, pd.DataFrame({"r": [s_next], "B_score": [y_next]})],
                             ignore_index=True)

        # persist iteration
        df_train.to_csv(os.path.join(data_dir, f"iter_{it:03d}.csv"), index=False)

        print(f"Iteration {it}: tried s={s_next:.4f}, observed B_score={y_next:.4f}")

    # final GP fit after loop
    X = df_train[["r"]].values
    y = df_train["B_score"].values
    gp = train_gp(X, y, length_scale=0.3, noise_level=0.01)

    # Produce plots
    plot_B_score_parity(df_train, df_test, gp, os.path.join(out_dir, "01_B_score_parity.png"))
    plot_B_score_confidence(df_train, df_test, gp, os.path.join(out_dir, "02_B_score_confidence.png"))
    plot_gradient_comparison_B(df_train, gp, os.path.join(out_dir, "03_gradient_comparison.png"))

    # simple optimization
    trajectory, B_trajectory = gradient_ascent_on_gp(gp, s0=0.4, lr=0.05, n_steps=20)
    plot_optimization_trajectory(df_train, gp, trajectory, B_trajectory,
                                 os.path.join(out_dir, "04_optimization_trajectory.png"))

    # Radar chart demo for Tanaka 5 indicators
    # generate radar demo data with non-monotonic components and s-dependent noise
    df_radar = make_radar_demo_df(
        seed=42,
        noise_scale=0.08,
        add_nonmonotonic=True,
        per_indicator_variability=True,
        reps=3,
    )
    try:
        plot_radar_grid(
            df_radar,
            s_col='s',
            ind_cols=('A', 'B', 'C', 'D', 'E', 'F'),
            b_col='B_score',
            cols_per_row=4,
            title='Tanaka indicators radar grid (static)',
            out_path=os.path.join(out_dir, '05_radar_grid.png'),
            normalize_mode='row',
            error_band=True,
        )
    except ImportError as e:
        print('Failed to create radar grid due to missing import:', e)
    except RuntimeError as e:
        print('Failed to create radar grid due to runtime error:', e)
    except Exception as e:
        print('Unexpected error during radar grid creation:', e)
        raise

    # Try to save an animation; if ffmpeg is not available, we skip saving.
    try:
        animate_radar(
            df_radar,
            s_col='s',
            ind_cols=('A', 'B', 'C', 'D', 'E', 'F'),
            b_col='B_score',
            title='Tanaka indicators radar animation',
            save_path=os.path.join(out_dir, '06_radar_animation.mp4'),
            fps=2,
        )
    except ImportError as e:
        print('Animation save skipped due to missing import (ffmpeg may be missing):', e)
    except RuntimeError as e:
        print('Animation save skipped due to runtime error (ffmpeg may be missing):', e)
    except Exception as e:
        print('Unexpected error during animation save:', e)
        raise

    print("Plots generated under:", out_dir)

    # --- Generate Tanaka demo CSV for use with tanaka_test tools ---
    try:
        # use training r values so we produce one row per trial
        r_train_vals = df_train['r'].values
        tanaka_df = make_tanaka_demo_full(r_train_vals, t0=1.0)
        tanaka_csv_path = os.path.join(out_dir, 'tanaka_demo.csv')
        tanaka_df.to_csv(tanaka_csv_path, index=False)
        print(f"Wrote Tanaka demo CSV (all trials) to: {tanaka_csv_path}")
    except Exception as e:
        print('Failed to write Tanaka demo CSV:', e)


if __name__ == "__main__":
    main()

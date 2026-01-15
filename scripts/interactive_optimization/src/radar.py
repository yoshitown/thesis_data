"""Radar chart helpers for Tanaka 5 indicators.

Provides functions to create a demo DataFrame, plot a grid of radar
charts and animate the transition over s.
All labels are English to avoid font/encoding issues.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter


def radar_angles(num_vars=6, start_angle=90):
    """Return angles for polar radar chart (radians)."""
    # Use the conventional radar setup: evenly spaced angles around the circle
    # (0..2pi), without duplicating the first point. The plotting function will
    # append the first angle when closing the polygon.
    angles = np.linspace(0.0, 2.0 * np.pi, num_vars, endpoint=False)
    return angles


def close_loop(values):
    return np.r_[values, values[0]]


def minmax_normalize(x, eps=1e-9):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + eps)


def normalize_indicators(df, cols, mode='global'):
    """Normalize indicator columns.

    mode:
      - 'global': min-max per column across the whole dataframe (default)
      - 'row': normalize each row by its row-wise max (keeps each sample relative)
    """
    # ensure cols is a list of column names (tuple would be interpreted
    # by pandas as a single multi-index key)
    cols = list(cols)
    out = df.copy()
    if mode == 'global':
        for c in cols:
            out[c] = minmax_normalize(out[c].values)
    elif mode == 'row':
        # for each row, divide by max across the indicator columns (like show_leder_chart_clean.normalize_df)
        vals = out[cols].values.astype(float)
        row_max = np.nanmax(vals, axis=1)
        # avoid division by zero
        safe_max = np.where(row_max == 0, np.nan, row_max)
        normed = (vals.T / safe_max).T
        normed = np.nan_to_num(normed, nan=0.0)
        out.loc[:, cols] = normed
    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")
    return out


def draw_single_radar(
    ax,
    values,
    angles,
    labels=None,
    color='C0',
    fill_alpha=0.25,
    rlim=(0, 1),
):
    vals = np.asarray(values, dtype=float)
    ax.set_ylim(*rlim)
    # Configure polar axes to match show_leder_chart_clean.py style
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # xticks use the base angles (no duplicate), and we append the first angle
    # for plotting to close the polygon.
    ax.set_xticks(angles)
    ax.set_yticks(np.linspace(rlim[0], rlim[1], 5))
    th = np.r_[angles, angles[0]]
    r = np.r_[vals, vals[0]]

    ax.plot(th, r, color=color, lw=2)
    ax.fill(th, r, color=color, alpha=fill_alpha)
    if labels is not None:
        # labels should match number of base angles
        ax.set_xticklabels(labels, fontsize=9)


def plot_radar_grid(
    df,
    s_col='s',
    ind_cols=('A', 'B', 'C', 'D', 'E', 'F'),
    b_col='B_score',
    cols_per_row=4,
    cmap='viridis',
    title='Tanaka indicators radar grid (static)',
    out_path=None,
    num_vertices: int = 6,
    normalize_mode: str = 'global',
    error_band: bool = True,
):
    """Plot a grid of radar charts (one per unique s).

    If out_path is provided, save the figure to that path; otherwise show.
    """
    labels = list(ind_cols)
    # support plotting as hexagon: if num_vertices != len(ind_cols), we will
    # append an auxiliary axis whose value is the mean of indicators
    num_vars = num_vertices
    angles = radar_angles(num_vars=num_vars, start_angle=90)
    # plotting limits for normalized indicators (used for error band clipping)
    rlim = (0.0, 1.0)

    uniq_s = np.sort(df[s_col].unique())
    n = len(uniq_s)
    rows = int(np.ceil(n / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, subplot_kw=dict(polar=True),
                             figsize=(4 * cols_per_row, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    df_norm = normalize_indicators(df, ind_cols, mode=normalize_mode)

    if b_col in df.columns:
        bvals = df.groupby(s_col)[b_col].mean().reindex(uniq_s).values
        bvals_n = minmax_normalize(bvals)
    else:
        bvals_n = np.zeros_like(uniq_s, dtype=float)
    cmap_obj = cm.get_cmap(cmap)

    for i, s in enumerate(uniq_s):
        ax = axes[i]
        group = df_norm[df_norm[s_col] == s][labels]
        # central tendency and dispersion
        base_vals = group.mean().values
        std_vals = group.std(ddof=0).values if len(group) > 1 else np.zeros_like(base_vals)
        # if num_vars > len(labels), append auxiliary value (mean)
        if num_vars > len(base_vals):
            aux = np.mean(base_vals)
            plot_vals = np.r_[base_vals, aux]
            plot_labels = labels + ['aux']
            std_vals = np.r_[std_vals, np.mean(std_vals) if std_vals.size > 0 else 0.0]
        else:
            plot_vals = base_vals
            plot_labels = labels

        color = cmap_obj(bvals_n[i]) if b_col in df.columns else 'C0'
        # draw mean polygon and optional error band
        th = np.r_[angles, angles[0]]
        r_mean = np.r_[plot_vals, plot_vals[0]]
        if error_band:
            r_low = np.clip(r_mean - np.r_[std_vals, std_vals[0]], *rlim)
            r_high = np.clip(r_mean + np.r_[std_vals, std_vals[0]], *rlim)
            # create closed polygon for band: high then reversed low
            th_band = np.r_[th, th[::-1]]
            r_band = np.r_[r_high, r_low[::-1]]
            ax.fill(th_band, r_band, color=color, alpha=0.15)

        draw_single_radar(
            ax,
            plot_vals,
            angles,
            labels=plot_labels,
            color=color,
            fill_alpha=0.35,
        )
        ax.set_title(f's = {s}', fontsize=11)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return out_path
    else:
        plt.show()


def animate_radar(
    df,
    s_col='s',
    ind_cols=('A', 'B', 'C', 'D', 'E', 'F'),
    b_col='B_score',
    cmap='viridis',
    title='Tanaka indicators radar animation',
    save_path=None,
    fps=2,
    num_vertices: int = 6,
    normalize_mode: str = 'global',
    error_band: bool = True,
):
    labels = list(ind_cols)
    num_vars = num_vertices
    angles = radar_angles(num_vars=num_vars, start_angle=90)

    uniq_s = np.sort(df[s_col].unique())
    df_norm = normalize_indicators(df, ind_cols, mode=normalize_mode)
    if b_col in df.columns:
        bvals = df.groupby(s_col)[b_col].mean().reindex(uniq_s).values
        bvals_n = minmax_normalize(bvals)
    else:
        bvals_n = np.zeros_like(uniq_s, dtype=float)
    cmap_obj = cm.get_cmap(cmap)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    title_txt = ax.set_title(title, fontsize=12, pad=20)

    line, = ax.plot([], [], lw=2)
    fill = None

    def init():
        nonlocal fill
        line.set_data([], [])
        if fill:
            fill.remove()
        return line,

    def update(frame):
        nonlocal fill
        s = uniq_s[frame]
        group = df_norm[df_norm[s_col] == s][labels]
        base_vals = group.mean().values
        std_vals = group.std(ddof=0).values if len(group) > 1 else np.zeros_like(base_vals)
        if num_vars > len(base_vals):
            aux = np.mean(base_vals)
            vals = np.r_[base_vals, aux]
            std_vals = np.r_[std_vals, np.mean(std_vals) if std_vals.size > 0 else 0.0]
        else:
            vals = base_vals

        th = np.r_[angles, angles[0]]
        r = close_loop(vals)
        color = cmap_obj(bvals_n[frame]) if b_col in df.columns else 'C0'
        line.set_data(th, r)
        line.set_color(color)
        if fill:
            fill.remove()
        if error_band:
            r_mean = r
            r_low = np.clip(r_mean - np.r_[std_vals, std_vals[0]], 0.0, 1.0)
            r_high = np.clip(r_mean + np.r_[std_vals, std_vals[0]], 0.0, 1.0)
            th_band = np.r_[th, th[::-1]]
            r_band = np.r_[r_high, r_low[::-1]]
            fill = ax.fill(th_band, r_band, color=color, alpha=0.15)[0]
        else:
            fill = ax.fill(th, r, color=color, alpha=0.35)[0]
        title_txt.set_text(f'{title}\ns = {s}')
        return line, fill

    ani = FuncAnimation(fig, update, frames=len(uniq_s), init_func=init,
                        blit=False, interval=1000 // fps)
    fig.tight_layout()

    if save_path:
        # mp4 saving requires ffmpeg. Request common options to maximize
        # QuickTime compatibility: use H.264 baseline/profile with yuv420p
        # pixel format and movflags for faststart.
        writer = FFMpegWriter(
            fps=fps,
            bitrate=1800,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'],
        )
        ani.save(save_path, writer=writer)
        plt.close(fig)
        return save_path
    else:
        plt.show()


def make_demo_df(
    seed: int = 42,
    noise_scale: float = 0.08,
    add_nonmonotonic: bool = True,
    per_indicator_variability: bool = True,
    reps: int = 3,
):
    """Generate demo dataframe with options to introduce variability.

    - `add_nonmonotonic`: add small sinusoidal components to break strict monotonic trend
    - `noise_scale`: base stddev for Gaussian noise; will be scaled with `s` to produce s-dependent noise
    - `per_indicator_variability`: apply small per-indicator multiplicative factor to simulate individual differences
    """
    rng = np.random.default_rng(seed)
    s_vals = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

    # base functional forms (same shapes as before)
    def base_A(s):
        return 2.0 + 0.5 * s + 0.3 * s ** 2

    def base_B(s):
        return 1.5 - 0.2 * s + 0.4 * s ** 2

    def base_C(s):
        return 1.0 + 0.8 * s - 0.3 * s ** 2

    def base_D(s):
        return 0.8 + 0.3 * s

    def base_E(s):
        return 0.5 + 0.1 * s + 0.2 * s ** 2

    def base_F(s):
        return 1.2 - 0.1 * s + 0.05 * s ** 2

    # per-indicator sinusoidal amplitudes/phases/frequencies
    amps = rng.uniform(0.05, 0.20, size=6) if add_nonmonotonic else np.zeros(6)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=6)
    freqs = rng.integers(1, 3, size=6)
    mults = rng.uniform(0.92, 1.08, size=6) if per_indicator_variability else np.ones(6)

    rows = []
    for s in s_vals:
        for _ in range(reps):
            # s-dependent noise: grows slightly with s
            noise_std = noise_scale * (0.5 + 0.5 * s)

            # base values
            A0 = base_A(s)
            B0 = base_B(s)
            C0 = base_C(s)
            D0 = base_D(s)
            E0 = base_E(s)
            F0 = base_F(s)

            # add small sinusoidal non-monotonic component per indicator
            def add_sin(idx, base):
                if not add_nonmonotonic:
                    return base
                return base + float(amps[idx] * np.sin(2.0 * np.pi * freqs[idx] * s + phases[idx]))

            A = add_sin(0, A0) * mults[0] + rng.normal(0.0, noise_std)
            Bv = add_sin(1, B0) * mults[1] + rng.normal(0.0, noise_std)
            C = add_sin(2, C0) * mults[2] + rng.normal(0.0, noise_std)
            D = add_sin(3, D0) * mults[3] + rng.normal(0.0, noise_std)
            E = add_sin(4, E0) * mults[4] + rng.normal(0.0, noise_std)
            F = add_sin(5, F0) * mults[5] + rng.normal(0.0, noise_std)

            # Calculate B_score as a weighted product of area, isotropy, symmetry, and centrality
            area = A
            isotropy = Bv
            symmetry = C
            centrality = D
            weights = [0.25, 0.25, 0.25, 0.25]
            B_score = (
                (area ** weights[0])
                * (isotropy ** weights[1])
                * (symmetry ** weights[2])
                * (centrality ** weights[3])
            )
            rows.append([s, A, Bv, C, D, E, F, B_score])

    df = pd.DataFrame(rows, columns=['s', 'A', 'B', 'C', 'D', 'E', 'F', 'B_score'])
    return df

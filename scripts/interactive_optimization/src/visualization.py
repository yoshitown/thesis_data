"""Plotting utilities for GP validation demo.

Implements parity, confidence, gradient comparison and
optimization trajectory plots as lightweight wrappers around
matplotlib.
"""
import os
import typing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_diagonal_line(ax, vals):
    min_val, max_val = vals.min() * 0.95, vals.max() * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.5,
            alpha=0.6, label='y=x')

def plot_B_score_parity(df_train, df_test, gp, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    y_pred_train = gp.predict(df_train[["r"]])
    y_pred_test = gp.predict(df_test[["r"]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        df_train["B_score"],
        y_pred_train,
        c="#3498db",
        s=100,
        alpha=0.7,
        edgecolor="white",
        linewidth=2,
        label="Train",
        marker="o",
    )
    ax.scatter(
        df_test["B_score"],
        y_pred_test,
        c="#e74c3c",
        s=100,
        alpha=0.7,
        edgecolor="white",
        linewidth=2,
        label="Test",
        marker="s",
    )

    all_vals = np.concatenate([df_train["B_score"], df_test["B_score"]])
    add_diagonal_line(ax, all_vals)

    r2_train = r2_score(df_train["B_score"], y_pred_train)
    r2_test = r2_score(df_test["B_score"], y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(df_test["B_score"], y_pred_test))

    ax.set_xlabel('Observed B(s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted B(s)', fontsize=14, fontweight='bold')
    ax.set_title(
        'GP prediction accuracy: B(s) score',
        fontsize=16,
        fontweight='bold',
        pad=20,
    )

    textstr = (
        f"R² (train) = {r2_train:.3f}\n"
        f"R² (test) = {r2_test:.3f}\n"
        f"RMSE (test) = {rmse_test:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="gray",
            linewidth=1.5,
        ),
    )

    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_B_score_confidence(df_train, df_test, gp, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    r_grid = np.linspace(0.35, 1.25, 100).reshape(-1, 1)
    y_pred, y_std = gp.predict(r_grid, return_std=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        r_grid.ravel(),
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        alpha=0.3,
        color="#95a5a6",
        label="95% confidence interval",
    )
    ax.plot(r_grid, y_pred, 'b-', linewidth=2.5, label='GP mean prediction')
    ax.scatter(
        df_train["r"],
        df_train["B_score"],
        c="#3498db",
        s=80,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
        label="Train",
        zorder=5,
    )
    ax.scatter(
        df_test["r"],
        df_test["B_score"],
        c="#e74c3c",
        s=80,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
        label="Test",
        marker="s",
        zorder=5,
    )

    ax.set_xlabel('s (crosslinker/monomer ratio)', fontsize=13, fontweight='bold')
    ax.set_ylabel('B(s) score', fontsize=13, fontweight='bold')
    ax.set_title(
        'GP surrogate predictive uncertainty',
        fontsize=15,
        fontweight='bold',
        pad=15,
    )
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gradient_comparison_B(df_train, gp, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    r_grid = np.linspace(0.4, 1.2, 50).reshape(-1, 1)
    delta = 1e-5
    grad_gp = (
        gp.predict(r_grid + delta) - gp.predict(r_grid - delta)
    ) / (2 * delta)

    r_unique = np.sort(df_train['r'].unique())
    B_mean = [df_train[df_train['r'] == r]['B_score'].mean() for r in r_unique]
    grad_fd = np.gradient(B_mean, r_unique)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        r_grid,
        grad_gp,
        'b-',
        linewidth=2.5,
        label='GP analytic gradient',
        alpha=0.9,
    )
    ax.plot(
        r_unique,
        grad_fd,
        'ro--',
        linewidth=2,
        markersize=8,
        label='Finite difference',
        alpha=0.7,
    )
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(
        's (crosslinker/monomer ratio)', fontsize=13, fontweight='bold'
    )
    ax.set_ylabel('dB/ds', fontsize=13, fontweight='bold')
    ax.set_title(
        'Gradient estimates: GP vs finite difference',
        fontsize=15,
        fontweight='bold',
        pad=15,
    )
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_optimization_trajectory(
    df_train, gp, trajectory, B_trajectory, out_path: str
):
    ensure_dir(os.path.dirname(out_path))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(B_trajectory, 'bo-', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('B(s)', fontsize=12, fontweight='bold')
    ax1.set_title('Optimization convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    r_grid = np.linspace(0.35, 1.25, 100).reshape(-1, 1)
    B_grid = gp.predict(r_grid)
    ax2.plot(
        r_grid,
        B_grid,
        'b-',
        linewidth=2.5,
        label='GP predicted B(s)',
        alpha=0.7,
    )
    ax2.scatter(
        df_train["r"],
        df_train["B_score"],
        c='gray',
        s=60,
        alpha=0.5,
        label='Train',
    )
    ax2.plot(
        trajectory,
        B_trajectory,
        'ro-',
        linewidth=2,
        markersize=8,
        label='Optimization path',
        zorder=5,
    )
    ax2.scatter(
        trajectory[0],
        B_trajectory[0],
        c='green',
        s=200,
        marker='*',
        label='Start',
        zorder=6,
    )
    ax2.scatter(
        trajectory[-1],
        B_trajectory[-1],
        c='red',
        s=200,
        marker='*',
        label='End',
        zorder=6,
    )
    ax2.set_xlabel(
        's (crosslinker/monomer ratio)', fontsize=12, fontweight='bold'
    )
    ax2.set_ylabel('B(s)', fontsize=12, fontweight='bold')
    ax2.set_title(
        'Optimization path visualization', fontsize=14,
        fontweight='bold'
    )
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

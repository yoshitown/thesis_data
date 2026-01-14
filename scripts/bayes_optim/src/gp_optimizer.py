"""Gaussian Process training, prediction and gradient helpers.

This module provides a small wrapper around sklearn's GaussianProcessRegressor
to train a 1D->1D surrogate and compute numerical gradients of the GP mean.
"""
from typing import Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def make_default_gp(length_scale: float = 0.3, noise_level: float = 1e-2):
    kernel = (
        RBF(length_scale=length_scale)
        + WhiteKernel(noise_level=noise_level)
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    return gp


def train_gp(X: np.ndarray, y: np.ndarray, length_scale: float = 0.3,
             noise_level: float = 1e-2) -> GaussianProcessRegressor:
    """Train and return a fitted GaussianProcessRegressor.

    X: shape (N, 1)
    y: shape (N,)
    """
    gp = make_default_gp(length_scale=length_scale, noise_level=noise_level)
    gp.fit(X, y)
    return gp


def gp_predict(gp: GaussianProcessRegressor, X: np.ndarray) -> np.ndarray:
    return gp.predict(X)


def gp_predict_with_std(gp: GaussianProcessRegressor, X: np.ndarray):
    return gp.predict(X, return_std=True)


def gp_mean_gradient_numeric(
    gp: GaussianProcessRegressor, X: np.ndarray, delta: float = 1e-5
) -> np.ndarray:
    """Compute numerical derivative of GP mean at points X using central diff.

    X: shape (M, 1)
    Returns shape (M,)
    """
    X = np.asarray(X).reshape(-1, 1)
    plus = gp.predict(X + delta)
    minus = gp.predict(X - delta)
    grad = (plus - minus) / (2.0 * delta)
    return grad


def finite_difference_gradient_from_samples(
    r_unique: np.ndarray, B_mean: np.ndarray
) -> np.ndarray:
    """Compute finite-difference gradient from sample points.

    r_unique: 1D sorted array of s values
    B_mean: 1D array of mean B(s) for each r_unique
    Returns gradient array same size as r_unique.
    """
    return np.gradient(B_mean, r_unique)


def gradient_ascent_on_gp(
    gp: GaussianProcessRegressor,
    s0: float,
    lr: float = 0.05,
    n_steps: int = 20,
    bounds: Tuple[float, float] = (0.4, 1.2),
) -> Tuple[list, list]:
    """Run simple gradient-ascent on a GP surrogate starting from s0.

    Returns (trajectory_s, trajectory_B)
    """
    s = float(s0)
    trajectory = [s]
    Bvals = [float(gp.predict(np.array([[s]]))[0])]

    for _ in range(n_steps):
        grad = gp_mean_gradient_numeric(gp, np.array([[s]]))[0]
        s = float(np.clip(s + lr * grad, bounds[0], bounds[1]))
        trajectory.append(s)
        Bvals.append(float(gp.predict(np.array([[s]]))[0]))

    return trajectory, Bvals


def suggest_next_point_discrete(
    gp: GaussianProcessRegressor,
    X_candidates: np.ndarray,
    beta: float = 1.96
) -> int:
    """Suggest the next point to query from a discrete set of candidates using UCB.

    X_candidates: shape (N, D)
    Returns the index of the suggested point in X_candidates.
    """
    mean, std = gp.predict(X_candidates, return_std=True)
    ucb = mean + beta * std
    return int(np.argmax(ucb))

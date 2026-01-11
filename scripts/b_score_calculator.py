"""B-score (B(s)) calculator utilities.

Implements a simplified version of the B(s) calculation from the spec.
"""
import numpy as np


def compute_B_score(H, S, A, B_indicator, C, alpha=1.0, beta=1.0, gamma=1.0, delta=0.5):
    """Compute the B(s) score from five Tanaka indicators.

    This is a pragmatic implementation following the spec's demo code.

    Inputs can be scalars or numpy arrays of the same shape. Returns numpy
    array or scalar accordingly.
    """
    indicators = np.array([H, S, A, B_indicator, C], dtype=float)

    # If indicators is shape (5, N) convert to (N, 5)
    if indicators.ndim == 2 and indicators.shape[0] == 5:
        indicators = indicators.T

    # Normalize each sample to 0-1 across the five indicators (demo heuristic)
    # Work on last axis
    mi = np.min(indicators, axis=-1, keepdims=True)
    ma = np.max(indicators, axis=-1, keepdims=True)
    denom = (ma - mi) + 1e-12
    normalized = (indicators - mi) / denom

    # area: mean of normalized indicators
    area = np.mean(normalized, axis=-1)

    # isotropy: CV penalty -> exp(-std / tau), tau=0.1 as demo
    isotropy = np.exp(-np.std(normalized, axis=-1) / 0.1)

    # symmetry: 1 / (1 + std) as demo
    symmetry = 1.0 / (1.0 + np.std(normalized, axis=-1))

    # centrality: penalty for mean away from 0.5
    centrality = np.exp(-np.abs(np.mean(normalized, axis=-1) - 0.5) / 0.2)

    B_s = (
        (area ** alpha)
        * (isotropy ** beta)
        * (symmetry ** gamma)
        * (centrality ** delta)
    )

    return B_s


def compute_B_from_matrix(mat, alpha=1.0, beta=1.0, gamma=1.0, delta=0.5):
    """Compute B(s) when given a matrix-like object with 5 indicators per row.

    mat: array-like with shape (N, 5) ordered as [H, S, A, B, C]
    Returns array shape (N,)
    """
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("mat must be shape (N,5)")
    return compute_B_score(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        arr[:, 3],  # B_indicator
        arr[:, 4],
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )

"""DTW + KDE utilities for IPD-style domain distance (Zhang et al. style).

Paired multivariate time series :math:`X, Y \\in \\mathbb{R}^{T \\times D}` use
per-step squared Euclidean cost and dynamic time warping. A Gaussian KDE is
then fit on the empirical per-pair distances (smooth bootstrap spirit from the
paper) and summarized by the mean of Monte Carlo draws, matching the pattern
in the authors' reference code (``KernelDensity`` + ``sample``).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.neighbors import KernelDensity


def _local_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean costs between time steps.

    Args:
        x: Array of shape ``(T1, D)``.
        y: Array of shape ``(T2, D)``.

    Returns:
        Cost matrix of shape ``(T1, T2)``.
    """
    # (T1, 1, D) - (1, T2, D) -> (T1, T2, D)
    diff = x[:, None, :] - y[None, :, :]
    return np.sum(diff * diff, axis=2)


def multivariate_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window_frac: Optional[float] = None,
) -> float:
    """Classic DTW distance with optional Sakoe–Chiba band.

    Args:
        x: Shape ``(T1, D)``.
        y: Shape ``(T2, D)``.
        window_frac: If set (e.g. ``0.5``), band width
            ``w = int(window_frac * max(T1, T2))`` so ``|i - j| <= w``.

    Returns:
        Square root of the minimal warping cost (L2 geometry on features).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Expected x (T1, D) and y (T2, D) with same D; got {x.shape}, {y.shape}."
        )
    t1, t2 = x.shape[0], y.shape[0]
    cost = _local_cost_matrix(x, y)
    inf = np.inf
    dp = np.full((t1 + 1, t2 + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    if window_frac is None:
        w = max(t1, t2)
    else:
        w = max(1, int(window_frac * max(t1, t2)))
    for i in range(1, t1 + 1):
        j_start = max(1, i - w)
        j_end = min(t2, i + w)
        for j in range(j_start, j_end + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]
            )
    return float(np.sqrt(dp[t1, t2]))


def batched_paired_dtw_distances(
    x: np.ndarray,
    y: np.ndarray,
    window_frac: Optional[float] = None,
) -> np.ndarray:
    """DTW for each aligned pair in a batch.

    Args:
        x: Shape ``(B, T, D)``.
        y: Shape ``(B, T, D)``.

    Returns:
        Distances of shape ``(B,)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must share shape; got {x.shape} vs {y.shape}.")
    b = x.shape[0]
    out = np.empty((b,), dtype=np.float64)
    for i in range(b):
        out[i] = multivariate_dtw_distance(x[i], y[i], window_frac=window_frac)
    return out


def kde_smoothed_scalar(
    distances: np.ndarray,
    *,
    bandwidth: float = 7.8,
    n_draws: int = 10,
    random_state: Union[None, int, np.random.Generator] = None,
) -> float:
    """Map empirical distances to a scalar via Gaussian KDE + mean of samples.

    Args:
        distances: 1D non-negative empirical inter-domain differences.
        bandwidth: KDE bandwidth (authors' reference used ``7.8``).
        n_draws: Number of KDE samples to average.
        random_state: Seed or ``Generator`` for reproducibility.

    Returns:
        Scalar summary; falls back to ``mean(distances)`` when KDE is undefined.
    """
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return 0.0
    if d.size == 1 or not np.isfinite(d).all():
        return float(np.mean(d))

    rs = random_state
    if isinstance(rs, np.random.Generator):
        seed_arg: Optional[int] = None
    else:
        seed_arg = int(rs) if rs is not None else None

    kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth)).fit(
        d.reshape(-1, 1)
    )
    # sklearn sample uses internal RNG; pass random_state when available
    try:
        samples = kde.sample(n_draws, random_state=seed_arg)
    except TypeError:
        samples = kde.sample(n_draws)
    return float(np.mean(samples))

"""
Relevant Lag Space Generation (RLSG).

Implements Algorithm 1 of the BBNNE paper. RLSG constructs `T` distinct
lag subspaces using binary tournament selection (k=2) with Pearson
correlation as the relevance metric.

Notes
-----
- Pearson correlation is computed using population standard deviation
  (ddof=0) to exactly match Eq. 2 of the paper.
- Lag indices are returned as 0-based indices into the lag matrix `X`.
  Lag index `i` corresponds to "lag (i+1)" in the paper's notation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def compute_relevance_scores(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson correlations and normalized relevance scores.

    Implements Eq. 2 (Pearson correlation) and Eq. 3 (relevance score)
    of the paper.

    Parameters
    ----------
    X : np.ndarray of shape (N, M)
        Lagged feature matrix (N samples, M candidate lags).
    y : np.ndarray of shape (N,)
        Target vector.

    Returns
    -------
    correlations : np.ndarray of shape (M,)
        Pearson correlation coefficient between each lag and the target.
    relevance_scores : np.ndarray of shape (M,)
        Normalized relevance scores RS_i = |r_i| / sum_j(|r_j|).
    """
    N, M = X.shape
    correlations = np.zeros(M, dtype=np.float64)

    y_mean = y.mean()
    y_std = y.std()  # population std (ddof=0)
    y_centered = y - y_mean

    for i in range(M):
        x = X[:, i]
        x_std = x.std()
        if x_std < 1e-12 or y_std < 1e-12:
            correlations[i] = 0.0
            continue
        x_centered = x - x.mean()
        # Eq. 2: Pearson correlation using population std
        correlations[i] = float(
            np.dot(x_centered, y_centered) / (N * x_std * y_std)
        )

    # Eq. 3: normalized relevance score
    denom = np.sum(np.abs(correlations))
    if denom < 1e-12:
        relevance_scores = np.full(M, 1.0 / M, dtype=np.float64)
    else:
        relevance_scores = np.abs(correlations) / denom

    return correlations, relevance_scores


def rlsg(
    X: np.ndarray,
    y: np.ndarray,
    n_spaces: int,
    subspace_size: int,
    seed: Optional[int] = None,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generate `n_spaces` distinct lag subspaces via binary tournament
    selection.

    Parameters
    ----------
    X : np.ndarray of shape (N, M)
        Lagged feature matrix.
    y : np.ndarray of shape (N,)
        Target vector.
    n_spaces : int
        Number of lag subspaces (equal to ensemble size T).
    subspace_size : int
        Number of lags in each subspace (denoted `tau` in the paper,
        but see NOTE below).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    lag_spaces : list of list of int
        One list of 0-based lag indices per subspace.
    relevance_scores : np.ndarray of shape (M,)

    NOTE
    ----
    The paper uses `tau` inconsistently:
      - Section 4.1: "candidate lag pool size (lag search space) to
        tau = 15".
      - Section 4.1: "feedforward neural network with a tau-10-1
        topology".
    We interpret `subspace_size` as the input dimension of each expert
    (empirically 5 based on Table 3), and the candidate pool size is
    handled separately in the caller (see `BBNNE.max_lag`).
    """
    if subspace_size > X.shape[1]:
        raise ValueError(
            f"subspace_size ({subspace_size}) cannot exceed the number of "
            f"candidate lags ({X.shape[1]})."
        )

    rng = np.random.default_rng(seed)
    _, relevance_scores = compute_relevance_scores(X, y)
    M = X.shape[1]

    # Edge case: the whole pool is requested as the subspace. There is
    # only one such subspace, so replicate it across all experts.
    if subspace_size == M:
        full_space = list(range(M))
        return [full_space[:] for _ in range(n_spaces)], relevance_scores

    lag_spaces: List[List[int]] = []
    for _ in range(n_spaces):
        available = list(range(M))
        space: List[int] = []
        for _ in range(subspace_size):
            # Binary tournament selection (k = 2)
            c1, c2 = rng.choice(available, size=2, replace=False)
            winner = (
                int(c1)
                if relevance_scores[c1] >= relevance_scores[c2]
                else int(c2)
            )
            space.append(winner)
            # Remove to prevent redundancy within the same subspace
            available.remove(winner)
        lag_spaces.append(space)

    return lag_spaces, relevance_scores
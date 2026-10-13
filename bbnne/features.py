"""
Rich feature engineering for BBNNE.

Modes
-----
"rich"     : 13 engineered features (current default behavior)
"raw_lags" : raw lagged prices, matching Eq. 1 of the paper

The default mode is "rich", so existing experiments are unaffected.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_RETURN_LAGS = (1, 2, 3, 5, 10)
DEFAULT_MA_WINDOWS = (5, 10, 20)
DEFAULT_VOL_WINDOWS = (5, 10, 20)
DEFAULT_RSI_WINDOWS = (5, 14)


def build_features(
    prices: np.ndarray,
    mode: str = "rich",
    return_lags: Sequence[int] = DEFAULT_RETURN_LAGS,
    ma_windows: Sequence[int] = DEFAULT_MA_WINDOWS,
    vol_windows: Sequence[int] = DEFAULT_VOL_WINDOWS,
    rsi_windows: Sequence[int] = DEFAULT_RSI_WINDOWS,
    n_lags: int = 15,
) -> tuple[np.ndarray, list[str]]:
    """
    Build features from a price series.

    Parameters
    ----------
    prices : np.ndarray
        Chronological price series.
    mode : {"rich", "raw_lags"}
        - "rich"     : the original 13 engineered features.
        - "raw_lags" : n_lags raw lagged prices, matching Eq. 1.
    n_lags : int
        Number of lags when mode="raw_lags". Ignored for mode="rich".

    Returns
    -------
    X : np.ndarray of shape (N, n_features)
    names : list of str
    """
    prices = np.asarray(prices, dtype=np.float64).ravel()
    N = len(prices)
    if N < 2:
        raise ValueError("Price series must have at least 2 samples.")

    # ---------- Mode 1: raw lagged prices (Eq. 1 of the paper) ----------
    if mode == "raw_lags":
        cols, names = [], []
        for i in range(1, n_lags + 1):
            col = np.empty(N, dtype=np.float64)
            col[:i] = prices[0]          # pad with the first observed price
            col[i:] = prices[:-i]
            cols.append(col)
            names.append(f"lag{i}")
        X = np.column_stack(cols).astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, names

    if mode != "rich":
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'rich' or 'raw_lags'."
        )

    # ---------- Mode 2: rich engineered features (original) ----------
    log_prices = np.log(np.maximum(prices, 1e-12))
    log_ret = np.diff(log_prices, prepend=log_prices[0])

    cols: list[np.ndarray] = []
    names: list[str] = []

    # 1. Lagged log-returns
    for L in return_lags:
        if L <= 0:
            raise ValueError(f"Return lag must be positive, got {L}.")
        col = np.empty(N, dtype=np.float64)
        col[:L] = 0.0
        col[L:] = log_ret[:-L]
        cols.append(col)
        names.append(f"logret_lag{L}")

    # 2. Distance from moving average
    s_prices = pd.Series(prices)
    for W in ma_windows:
        ma = s_prices.rolling(W, min_periods=1).mean().to_numpy()
        col = prices / np.maximum(ma, 1e-12) - 1.0
        cols.append(col)
        names.append(f"ma_dist{W}")

    # 3. Rolling volatility
    s_ret = pd.Series(log_ret)
    for W in vol_windows:
        vol = s_ret.rolling(W, min_periods=1).std().to_numpy()
        vol = np.nan_to_num(vol, nan=0.0)
        cols.append(vol)
        names.append(f"vol{W}")

    # 4. RSI-style momentum
    up = np.maximum(log_ret, 0.0)
    dn = np.maximum(-log_ret, 0.0)
    s_up = pd.Series(up)
    s_dn = pd.Series(dn)
    for W in rsi_windows:
        avg_up = s_up.rolling(W, min_periods=1).mean().to_numpy()
        avg_dn = s_dn.rolling(W, min_periods=1).mean().to_numpy()
        rsi = avg_up / (avg_up + avg_dn + 1e-12)
        cols.append(rsi)
        names.append(f"rsi{W}")

    X = np.column_stack(cols).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names
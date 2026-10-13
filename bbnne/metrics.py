"""
Evaluation metrics for time series forecasting.

Implements:
    - MAPE, MAE, RMSE
    - Directional Accuracy (Eq. 14)
    - Theil's U statistic (Eq. 15) — HORIZON-AWARE
    - R² (coefficient of determination)

HORIZON-AWARE THEIL'S U
-----------------------
The classical Theil's U compares the model's RMSE against a naive
persistence forecast on the SAME horizon. When the model predicts an
h-step-ahead target, both numerator and denominator must be evaluated
on the h-step scale; otherwise the statistic is inflated by roughly
sqrt(h). This module therefore requires an explicit `horizon`
argument in `theils_u` and `evaluate`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

_EPS = 1e-12


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.where(np.abs(y_true) < _EPS, _EPS, np.abs(y_true))
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / denom))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (%).

    Fraction of time steps at which the predicted and true one-step
    price changes share the same sign (Eq. 14 of the paper).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return float("nan")

    true_change = y_true[1:] - y_true[:-1]
    pred_change = y_pred[1:] - y_true[:-1]

    valid = (np.abs(true_change) > _EPS) & (np.abs(pred_change) > _EPS)
    if valid.sum() == 0:
        return float("nan")

    same_sign = np.sign(true_change[valid]) == np.sign(pred_change[valid])
    return float(100.0 * np.mean(same_sign))


def theils_u(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> float:
    """
    Horizon-aware Theil's U statistic.

    .. math::
        U = \\frac{\\sqrt{\\frac{1}{N}\\sum_t (Y_t - \\hat Y_t)^2}}
                 {\\sqrt{\\frac{1}{N-h}\\sum_t (Y_{t} - Y_{t-h})^2}}

    The numerator is the RMSE of the model's h-step-ahead prediction
    errors. The denominator is the RMSE of the h-step-ahead price
    changes that a naive persistence forecast would fail to explain.
    Both terms are therefore on the same temporal scale. A value of
    U < 1 indicates that the model beats the h-step persistence
    baseline; U >= 1 indicates that it does not.

    Parameters
    ----------
    y_true : np.ndarray
        Aligned target prices. y_true[t] is the price observed h steps
        after the anchor used by the model at test position t.
    y_pred : np.ndarray
        Model predictions aligned with `y_true`.
    horizon : int
        Forecasting horizon h in steps. Must match the horizon used to
        generate `y_pred`. Default is 1.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length; "
            f"got {len(y_true)} and {len(y_pred)}."
        )
    if len(y_true) <= h:
        return float("nan")

    num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    den = np.sqrt(np.mean((y_true[h:] - y_true[:-h]) ** 2))
    if den < _EPS:
        return float("nan")
    return float(num / den)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination R².

    .. math::
        R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}
                       {\\sum (y_i - \\bar{y})^2}
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < _EPS:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics in one call.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Aligned actual and predicted prices.
    horizon : int
        Forecasting horizon h in steps. Forwarded to `theils_u`.
    """
    return {
        "MAPE(%)": mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "DA(%)": directional_accuracy(y_true, y_pred),
        "TheilU": theils_u(y_true, y_pred, horizon=horizon),
        "R2": r_squared(y_true, y_pred),
    }
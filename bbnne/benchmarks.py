"""
Baseline forecasting models with the SAME feature pipeline and target
as BBNNE, for a fair comparison.

Target: h-step ahead log-return  r_h(k) = log(P(k+h) / P(k))
Final prediction:  y_hat(k+h) = P(k) * exp(gamma * r_h_hat(k))
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ------------------------------------------------------------------ #
# Shared data preparation
# ------------------------------------------------------------------ #
def _prepare_xy(
    prices: np.ndarray,
    max_lag: int,
    horizon: int = 1,
    feature_mode: str = "rich",
):
    """
    Build (features, anchors, h-step log-return target) aligned to the
    same rows used by BBNNE._build_dataset.
    """
    F, _ = build_features(prices, mode=feature_mode, n_lags=max_lag)
    N = len(prices)
    L = max_lag
    h = horizon
    log_prices = np.log(np.maximum(prices, 1e-12))

    start = L - 1
    end = N - h
    X = F[start:end]
    anchors = prices[start:end]
    targets = log_prices[start + h: N] - log_prices[start:end]
    return X, anchors, targets


def _predict_prices(
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    anchors_te: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Fit on standardized log-returns, predict back to price level."""
    feat_scaler = StandardScaler().fit(X_tr)
    X_tr_s = feat_scaler.transform(X_tr)
    X_te_s = feat_scaler.transform(X_te)

    tgt_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
    y_tr_s = tgt_scaler.transform(y_tr.reshape(-1, 1)).ravel()

    model.fit(X_tr_s, y_tr_s)
    r_pred_s = model.predict(X_te_s)
    r_pred = tgt_scaler.inverse_transform(
        r_pred_s.reshape(-1, 1)
    ).ravel()
    return anchors_te * np.exp(shrinkage * r_pred)


# ------------------------------------------------------------------ #
# Baseline 1: MLP
# ------------------------------------------------------------------ #
def fit_mlp(
    prices_train: np.ndarray,
    prices_test: np.ndarray,
    max_lag: int = 15,
    hidden: int = 6,
    seed: int = 42,
    shrinkage: float = 0.5,
    horizon: int = 1,
    feature_mode: str = "rich",
) -> np.ndarray:
    X_tr, _, y_tr = _prepare_xy(
        prices_train, max_lag, horizon, feature_mode
    )
    X_te, anchors_te, _ = _prepare_xy(
        prices_test, max_lag, horizon, feature_mode
    )

    model = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="logistic",
        solver="lbfgs",
        max_iter=2000,
        tol=1e-6,
        random_state=seed,
    )
    return _predict_prices(
        model, X_tr, y_tr, X_te, anchors_te, shrinkage=shrinkage
    )


# ------------------------------------------------------------------ #
# Baseline 2: RBF (Random Fourier Features + Ridge)
# ------------------------------------------------------------------ #
def fit_rbf(
    prices_train: np.ndarray,
    prices_test: np.ndarray,
    max_lag: int = 15,
    n_centers: int = 6,
    n_features: int = 32,
    gamma: float = 1.0,
    seed: int = 42,
    shrinkage: float = 0.5,
    horizon: int = 1,
    feature_mode: str = "rich",
) -> np.ndarray:
    X_tr, _, y_tr = _prepare_xy(
        prices_train, max_lag, horizon, feature_mode
    )
    X_te, anchors_te, _ = _prepare_xy(
        prices_test, max_lag, horizon, feature_mode
    )

    model = Pipeline([
        ("rbf", RBFSampler(
            gamma=gamma, n_components=n_features, random_state=seed
        )),
        ("ridge", Ridge(alpha=1e-3)),
    ])
    return _predict_prices(
        model, X_tr, y_tr, X_te, anchors_te, shrinkage=shrinkage
    )


# ------------------------------------------------------------------ #
# Baseline 3: Bagging of MLPs
# ------------------------------------------------------------------ #
def fit_bagging(
    prices_train: np.ndarray,
    prices_test: np.ndarray,
    max_lag: int = 15,
    n_estimators: int = 7,
    hidden: int = 6,
    seed: int = 42,
    shrinkage: float = 0.5,
    horizon: int = 1,
    feature_mode: str = "rich",
) -> np.ndarray:
    X_tr, _, y_tr = _prepare_xy(
        prices_train, max_lag, horizon, feature_mode
    )
    X_te, anchors_te, _ = _prepare_xy(
        prices_test, max_lag, horizon, feature_mode
    )

    base = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="logistic",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    model = BaggingRegressor(
        estimator=base,
        n_estimators=n_estimators,
        random_state=seed,
    )
    return _predict_prices(
        model, X_tr, y_tr, X_te, anchors_te, shrinkage=shrinkage
    )


# ------------------------------------------------------------------ #
# Baseline 4: AdaBoost.R2
# ------------------------------------------------------------------ #
def fit_adaboost(
    prices_train: np.ndarray,
    prices_test: np.ndarray,
    max_lag: int = 15,
    n_estimators: int = 5,
    hidden: int = 6,
    seed: int = 42,
    shrinkage: float = 0.5,
    horizon: int = 1,
    feature_mode: str = "rich",
) -> np.ndarray:
    X_tr, _, y_tr = _prepare_xy(
        prices_train, max_lag, horizon, feature_mode
    )
    X_te, anchors_te, _ = _prepare_xy(
        prices_test, max_lag, horizon, feature_mode
    )

    base = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="logistic",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    model = AdaBoostRegressor(
        estimator=base,
        n_estimators=n_estimators,
        random_state=seed,
    )
    return _predict_prices(
        model, X_tr, y_tr, X_te, anchors_te, shrinkage=shrinkage
    )


# ------------------------------------------------------------------ #
# Baseline 5: LSTM (optional, requires TensorFlow)
# ------------------------------------------------------------------ #
def fit_lstm(
    prices_train: np.ndarray,
    prices_test: np.ndarray,
    max_lag: int = 15,
    hidden: int = 20,
    epochs: int = 200,
    patience: int = 15,
    dropout: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
    shrinkage: float = 0.5,
    horizon: int = 1,
    feature_mode: str = "rich",
) -> Optional[np.ndarray]:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
    except ImportError:
        return None

    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_tr, _, y_tr = _prepare_xy(
        prices_train, max_lag, horizon, feature_mode
    )
    X_te, anchors_te, _ = _prepare_xy(
        prices_test, max_lag, horizon, feature_mode
    )

    feat_scaler = StandardScaler().fit(X_tr)
    X_tr_s = feat_scaler.transform(X_tr)
    X_te_s = feat_scaler.transform(X_te)
    tgt_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
    y_tr_s = tgt_scaler.transform(y_tr.reshape(-1, 1)).ravel()

    X_tr_r = X_tr_s.reshape(-1, 1, X_tr_s.shape[1])
    X_te_r = X_te_s.reshape(-1, 1, X_te_s.shape[1])

    model = models.Sequential([
        layers.Input(shape=(1, X_tr_s.shape[1])),
        layers.LSTM(hidden, activation="tanh"),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    es = callbacks.EarlyStopping(
        monitor="loss", patience=patience, restore_best_weights=True
    )
    model.fit(X_tr_r, y_tr_s, epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=[es])

    r_pred_s = model.predict(X_te_r, verbose=0).ravel()
    r_pred = tgt_scaler.inverse_transform(
        r_pred_s.reshape(-1, 1)
    ).ravel()
    return anchors_te * np.exp(shrinkage * r_pred)
"""
BBNNE: Bat-Based Boosted Neural Network Ensemble with RLSG.

Target (h-step ahead):  r_h(k) = log(P(k+h) / P(k))
Prediction:             y_hat(k+h) = P(k) * exp(gamma * r_h_hat(k))

Sample weighting follows Algorithm 2 of the paper using the Absolute
Percentage Error (APE) on the PRICE level, i.e.

    APE_i = |P_i - P_hat_i| / |P_i| = |1 - exp(r_hat_i - r_i)|

with a dynamic, iteration-dependent threshold φ_t equal to the mean
APE of the current expert. This matches the manuscript exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from .features import build_features
from .neural_network import MLPExpert
from .rlsg import rlsg


class BBNNE:
    def __init__(
        self,
        n_experts: int = 7,
        max_lag: int = 15,
        subspace_size: int = 5,
        hidden_size: int = 10,
        bat_params: Optional[Dict] = None,
        seed: Optional[int] = None,
        sampling_seed: Optional[int] = None,
        residual_shrinkage: float = 0.5,
        use_boosting: bool = True,
        horizon: int = 1,
        feature_mode: str = "rich",
    ) -> None:
        self.n_experts = int(n_experts)
        self.max_lag = int(max_lag)
        self.subspace_size = int(subspace_size)
        self.hidden_size = int(hidden_size)
        self.bat_params = bat_params or {}
        self.seed = seed
        self.sampling_seed = (
            sampling_seed if sampling_seed is not None else seed
        )
        self.residual_shrinkage = float(residual_shrinkage)
        self.use_boosting = bool(use_boosting)
        self.horizon = int(horizon)
        self.feature_mode = str(feature_mode)

        if self.horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if self.feature_mode not in ("rich", "raw_lags"):
            raise ValueError(
                f"feature_mode must be 'rich' or 'raw_lags', "
                f"got '{self.feature_mode}'."
            )

        self.rng = np.random.default_rng(self.sampling_seed)
        self.lag_spaces: Optional[List[List[int]]] = None
        self.experts: List[MLPExpert] = []
        self.expert_weights: Optional[np.ndarray] = None
        self.beta_values: List[float] = []
        self.phi_history: List[float] = []
        self.feature_scaler_: Optional[StandardScaler] = None
        self.target_scaler_: Optional[StandardScaler] = None
        self.n_experts_fitted_: int = 0

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    def _build_dataset(self, prices: np.ndarray):
        F, _ = build_features(
            prices, mode=self.feature_mode, n_lags=self.max_lag
        )
        L = self.max_lag
        h = self.horizon
        N = len(prices)
        log_prices = np.log(np.maximum(prices, 1e-12))

        start = L - 1
        end = N - h
        X = F[start:end]
        anchors = prices[start:end]
        targets = log_prices[start + h: N] - log_prices[start:end]
        return X, anchors, targets

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, prices: np.ndarray, verbose: bool = False) -> "BBNNE":
        prices = np.asarray(prices, dtype=np.float64).ravel()
        X_raw, anchors, targets = self._build_dataset(prices)
        N = X_raw.shape[0]

        self.feature_scaler_ = StandardScaler().fit(X_raw)
        X = self.feature_scaler_.transform(X_raw)
        self.target_scaler_ = StandardScaler().fit(targets.reshape(-1, 1))
        y = self.target_scaler_.transform(
            targets.reshape(-1, 1)
        ).ravel()

        if verbose:
            print(f"[Data] N={N} | n_features={X.shape[1]} | "
                  f"T={self.n_experts} | h={self.horizon} | "
                  f"mode={self.feature_mode}")

        self.lag_spaces, self.relevance_scores = rlsg(
            X, y,
            n_spaces=self.n_experts,
            subspace_size=self.subspace_size,
            seed=self.seed,
        )

        if self.use_boosting:
            w = np.full(N, 1.0 / N, dtype=np.float64)
        self.experts = []
        self.beta_values = []
        self.phi_history = []

        for t in range(self.n_experts):
            if self.use_boosting:
                p = w / w.sum()
                idx = self.rng.choice(N, size=N, replace=True, p=p)
            else:
                idx = self.rng.integers(0, N, size=N)

            X_sub = X[np.ix_(idx, self.lag_spaces[t])]
            y_sub = y[idx]

            expert = MLPExpert(
                n_input=self.subspace_size,
                n_hidden=self.hidden_size,
                seed=self.seed,
            )
            expert.fit(
                X_sub, y_sub, bat_params=self.bat_params, verbose=False
            )
            self.experts.append(expert)

            if self.use_boosting:
                # Expert prediction in standardized target space
                y_pred_s = expert.predict(X[:, self.lag_spaces[t]])
                y_pred_real = self.target_scaler_.inverse_transform(
                    y_pred_s.reshape(-1, 1)
                ).ravel()

                # --------------------------------------------------------
                # APE on the price level (Algorithm 2 in the paper):
                #   APE_i = |P_i - P_hat_i| / |P_i|
                #         = |1 - exp(r_hat_i - r_i)|
                # This is algebraically identical and numerically stable.
                # --------------------------------------------------------
                ape = np.abs(1.0 - np.exp(y_pred_real - targets))

                # Dynamic, iteration-dependent threshold (Eq. in Sec. 3.2)
                phi_t = float(ape.mean())
                self.phi_history.append(phi_t)

                # Weighted error rate
                epsilon_t = float(np.sum(p * (ape > phi_t)))
                epsilon_t = float(
                    np.clip(epsilon_t, 1e-6, 0.5 - 1e-6)
                )
                beta_t = epsilon_t / (1.0 - epsilon_t)
                self.beta_values.append(beta_t)

                # Down-weight easy samples (APE <= phi_t)
                update_mask = ape <= phi_t
                w = np.where(update_mask, w * beta_t, w)
                w = w / w.sum()

        if self.use_boosting:
            log_inv_beta = np.log(1.0 / np.array(self.beta_values))
            denom = log_inv_beta.sum()
            if denom <= 0:
                self.expert_weights = np.full(
                    len(self.experts), 1.0 / len(self.experts)
                )
            else:
                self.expert_weights = log_inv_beta / denom
        else:
            self.expert_weights = np.full(
                len(self.experts), 1.0 / len(self.experts)
            )

        self.n_experts_fitted_ = len(self.experts)
        if verbose:
            print(f"[Fitted] {self.n_experts_fitted_} experts | "
                  f"gamma={self.residual_shrinkage} | "
                  f"phi_t range=[{min(self.phi_history):.4g}, "
                  f"{max(self.phi_history):.4g}]")
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64).ravel()
        X_raw, anchors, _ = self._build_dataset(prices)
        X = self.feature_scaler_.transform(X_raw)

        y_pred_s = np.zeros(len(X), dtype=np.float64)
        for t, expert in enumerate(self.experts):
            y_pred_s += self.expert_weights[t] * expert.predict(
                X[:, self.lag_spaces[t]]
            )
        r_pred = self.target_scaler_.inverse_transform(
            y_pred_s.reshape(-1, 1)
        ).ravel()
        return anchors * np.exp(self.residual_shrinkage * r_pred)
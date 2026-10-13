"""
Ablation study for BBNNE.

Design
------
The APE-thresholded boosting mechanism is the CORE of BBNNE and is
therefore kept ACTIVE in every variant. The ablation varies only the
two remaining axes:

    - RLSG   : feature-space diversity (diverse lag subspaces)
    - Trainer: metaheuristic training (Bat) vs. gradient-based (BP)

Variants
--------
"full"          : RLSG + Bat  + Boosting      (complete BBNNE)
"wo_rlsg"       : full features + Bat + Boosting
"wo_bat"        : RLSG + BP + Boosting
"wo_rlsg_bat"   : full features + BP + Boosting  (optional)

Boosting is enabled in all four variants.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from .features import build_features
from .neural_network import MLPExpert
from .rlsg import rlsg


_VALID_VARIANTS = ("full", "wo_rlsg", "wo_bat", "wo_rlsg_bat")


class BBNNEAblation:
    """
    BBNNE with switchable RLSG and trainer, for the ablation study.

    Boosting is always active.
    """

    def __init__(
        self,
        n_experts: int = 7,
        max_lag: int = 15,
        subspace_size: int = 5,
        hidden_size: int = 10,
        bat_params: Optional[Dict] = None,
        seed: Optional[int] = None,
        sampling_seed: Optional[int] = None,
        variant: str = "full",
        residual_shrinkage: float = 0.5,
        horizon: int = 1,
        feature_mode: str = "rich",
    ) -> None:
        if variant not in _VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {_VALID_VARIANTS}, "
                f"got '{variant}'."
            )

        self.n_experts = int(n_experts)
        self.max_lag = int(max_lag)
        self.subspace_size = int(subspace_size)
        self.hidden_size = int(hidden_size)
        self.bat_params = bat_params or {}
        self.seed = seed
        self.sampling_seed = (
            sampling_seed if sampling_seed is not None else seed
        )
        self.variant = str(variant)
        self.residual_shrinkage = float(residual_shrinkage)
        self.horizon = int(horizon)
        self.feature_mode = str(feature_mode)

        if self.horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if self.feature_mode not in ("rich", "raw_lags"):
            raise ValueError(
                f"feature_mode must be 'rich' or 'raw_lags', "
                f"got '{self.feature_mode}'."
            )

        # ---- Ablation switches ----
        # Boosting is ALWAYS active (core of BBNNE).
        self.rlsg_enabled = self.variant in ("full", "wo_bat")
        self.bat_enabled = self.variant in ("full", "wo_rlsg")
        self.boosting_enabled = True

        self.rng = np.random.default_rng(self.sampling_seed)

        self.lag_spaces: Optional[List[List[int]]] = None
        self.experts: List[MLPExpert] = []
        self.expert_weights: Optional[np.ndarray] = None
        self.beta_values: List[float] = []
        self.phi_history: List[float] = []
        self.input_dim_per_expert_: int = 0
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
    # Subspaces
    # ------------------------------------------------------------------ #
    def _generate_lag_spaces(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.rlsg_enabled:
            # RLSG: each expert gets its own diverse subspace
            self.lag_spaces, _ = rlsg(
                X, y,
                n_spaces=self.n_experts,
                subspace_size=self.subspace_size,
                seed=self.seed,
            )
            self.input_dim_per_expert_ = self.subspace_size
        else:
            # No RLSG: every expert uses the full feature matrix
            full = list(range(X.shape[1]))
            self.lag_spaces = [full[:] for _ in range(self.n_experts)]
            self.input_dim_per_expert_ = X.shape[1]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self, prices: np.ndarray, verbose: bool = False
    ) -> "BBNNEAblation":
        prices = np.asarray(prices, dtype=np.float64).ravel()
        X_raw, anchors, targets = self._build_dataset(prices)
        N = X_raw.shape[0]

        self.feature_scaler_ = StandardScaler().fit(X_raw)
        X = self.feature_scaler_.transform(X_raw)
        self.target_scaler_ = StandardScaler().fit(targets.reshape(-1, 1))
        y = self.target_scaler_.transform(
            targets.reshape(-1, 1)
        ).ravel()

        self._generate_lag_spaces(X, y)

        # Boosting is always active
        use_boosting = self.boosting_enabled
        effective_trainer = "bat" if self.bat_enabled else "bp"

        if verbose:
            print(f"[Variant] {self.variant_label()}")
            print(f"  n_experts={self.n_experts} | "
                  f"input_dim={self.input_dim_per_expert_} | "
                  f"rlsg={self.rlsg_enabled} | "
                  f"trainer={effective_trainer} | "
                  f"boosting={use_boosting} | N={N} | "
                  f"h={self.horizon} | mode={self.feature_mode}")

        w = np.full(N, 1.0 / N, dtype=np.float64)
        self.experts = []
        self.beta_values = []
        self.phi_history = []

        for t in range(self.n_experts):
            # ---- Sampling with boosting weights ----
            p = w / w.sum()
            idx = self.rng.choice(N, size=N, replace=True, p=p)

            X_sub = X[np.ix_(idx, self.lag_spaces[t])]
            y_sub = y[idx]

            # ---- Train expert with the selected trainer ----
            expert = MLPExpert(
                n_input=self.input_dim_per_expert_,
                n_hidden=self.hidden_size,
                seed=self.seed,
            )
            expert.fit(
                X_sub, y_sub,
                bat_params=self.bat_params,
                verbose=False,
                trainer=effective_trainer,
            )
            self.experts.append(expert)

            # ---- Boosting weight update (APE on price level) ----
            y_pred_s = expert.predict(X[:, self.lag_spaces[t]])
            y_pred_real = self.target_scaler_.inverse_transform(
                y_pred_s.reshape(-1, 1)
            ).ravel()

            # APE on price level (matches Algorithm 2 of the paper)
            ape = np.abs(1.0 - np.exp(y_pred_real - targets))

            phi_t = float(ape.mean())
            self.phi_history.append(phi_t)

            epsilon_t = float(np.sum(p * (ape > phi_t)))
            epsilon_t = float(np.clip(epsilon_t, 1e-6, 0.5 - 1e-6))
            beta_t = epsilon_t / (1.0 - epsilon_t)
            self.beta_values.append(beta_t)

            # Down-weight easy samples
            update_mask = ape <= phi_t
            w = np.where(update_mask, w * beta_t, w)
            w = w / w.sum()

        # ---- Final expert weights (log(1/beta), normalised) ----
        if len(self.beta_values) > 0:
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
            print(f"  Fitted {self.n_experts_fitted_} experts | "
                  f"weights={self.expert_weights.round(4)}")
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, prices: np.ndarray) -> np.ndarray:
        if self.expert_weights is None:
            raise RuntimeError("Model has not been fitted.")

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

    # ------------------------------------------------------------------ #
    def variant_label(self) -> str:
        return {
            "full":         "Full BBNNE",
            "wo_rlsg":      "BBNNE w/o RLSG",
            "wo_bat":       "BBNNE w/o Bat",
            "wo_rlsg_bat":  "BBNNE w/o RLSG & Bat",
        }.get(self.variant, f"BBNNE [{self.variant}]")
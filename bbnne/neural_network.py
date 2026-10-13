"""
MLP expert with a single hidden layer (tau -> hidden -> 1).

Activation
----------
- Hidden layer : log-sigmoid   f(z) = 1 / (1 + exp(-z))
- Output layer : linear

Weight optimization is delegated to one of two trainers:
    - "bat" : Bat Algorithm (metaheuristic, default)
    - "bp"  : Backpropagation via sklearn MLPRegressor (L-BFGS)

The `trainer` argument is used by the ablation study to isolate the
contribution of the Bat Algorithm from the rest of the framework.

Note
----
The BP trainer may emit a ConvergenceWarning when L-BFGS reaches its
iteration limit. This warning is suppressed here because it is
informational only: the network still returns the best weights found,
and the surrounding ensemble is unaffected. If higher precision is
needed, increase `max_iter` in `fit()`.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
from scipy.special import expit  # numerically stable sigmoid

from .bat_algorithm import BatAlgorithm


class MLPExpert:
    """
    Single-hidden-layer feedforward neural network.

    Parameters
    ----------
    n_input : int
        Number of input features (equal to the subspace size tau).
    n_hidden : int
        Number of hidden units (default 10, per the paper's topology).
    n_output : int
        Number of output units (always 1 for regression here).
    seed : int, optional
        Random seed for the Bat Algorithm.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 10,
        n_output: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.n_input = int(n_input)
        self.n_hidden = int(n_hidden)
        self.n_output = int(n_output)
        self.seed = seed

        # Total number of trainable parameters
        self.n_params = (
            self.n_input * self.n_hidden + self.n_hidden
            + self.n_hidden * self.n_output + self.n_output
        )

        # ---- Runtime attributes ----
        self.weights: Optional[np.ndarray] = None
        self.train_mse: Optional[float] = None
        self.optimizer_history: Optional[Dict] = None
        self.trainer_: str = "bat"
        self._bp_model = None  # only set when trainer == "bp"

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #
    def _unpack(self, w: np.ndarray):
        """Split a flat parameter vector into (W1, b1, W2, b2)."""
        idx = 0
        W1 = w[idx: idx + self.n_input * self.n_hidden].reshape(
            self.n_input, self.n_hidden
        )
        idx += self.n_input * self.n_hidden
        b1 = w[idx: idx + self.n_hidden]
        idx += self.n_hidden
        W2 = w[idx: idx + self.n_hidden * self.n_output].reshape(
            self.n_hidden, self.n_output
        )
        idx += self.n_hidden * self.n_output
        b2 = w[idx: idx + self.n_output]
        return W1, b1, W2, b2

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Vectorized forward pass through the network."""
        W1, b1, W2, b2 = self._unpack(w)
        z1 = X @ W1 + b1
        a1 = expit(z1)             # log-sigmoid, numerically stable
        z2 = a1 @ W2 + b2          # linear output
        return z2.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new inputs (requires a trained model)."""
        if self.trainer_ == "bp":
            if self._bp_model is None:
                raise RuntimeError(
                    "MLPExpert (BP trainer) has not been trained yet."
                )
            return self._bp_model.predict(X)
        if self.weights is None:
            raise RuntimeError("MLPExpert has not been trained yet.")
        return self.forward(X, self.weights)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bat_params: Optional[Dict] = None,
        verbose: bool = False,
        trainer: str = "bat",
    ) -> "MLPExpert":
        """
        Train network weights.

        Parameters
        ----------
        X : np.ndarray of shape (N, n_input)
        y : np.ndarray of shape (N,)
        bat_params : dict, optional
            Hyperparameters for the Bat Algorithm (only used when
            trainer == "bat").
        verbose : bool
            If True, the Bat Algorithm prints progress.
        trainer : {"bat", "bp"}
            - "bat" : Bat Algorithm (default, as in the paper).
            - "bp"  : Backpropagation via sklearn's MLPRegressor
                      (L-BFGS solver). Used in the ablation study to
                      isolate the contribution of the Bat Algorithm.

        Returns
        -------
        self
        """
        if trainer not in ("bat", "bp"):
            raise ValueError(
                f"trainer must be 'bat' or 'bp', got '{trainer}'."
            )

        # ------------------------------------------------------------ #
        # Trainer: Backpropagation (L-BFGS)
        # ------------------------------------------------------------ #
        if trainer == "bp":
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.neural_network import MLPRegressor

            self._bp_model = MLPRegressor(
                hidden_layer_sizes=(self.n_hidden,),
                activation="logistic",      # log-sigmoid
                solver="lbfgs",             # quasi-Newton
                max_iter=2000,
                tol=1e-6,
                random_state=self.seed,
            )
            # The ConvergenceWarning emitted by sklearn when L-BFGS
            # reaches the iteration limit is informational only: the
            # model still returns the best weights found during
            # optimisation, and the ensemble is unaffected. We suppress
            # it here to keep the console output clean.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=ConvergenceWarning
                )
                self._bp_model.fit(X, y)
            self.trainer_ = "bp"
            return self

        # ------------------------------------------------------------ #
        # Trainer: Bat Algorithm (default)
        # ------------------------------------------------------------ #
        if bat_params is None:
            bat_params = {}

        optimizer = BatAlgorithm(
            n_params=self.n_params,
            seed=self.seed,
            **bat_params,
        )

        def fitness(w: np.ndarray) -> float:
            y_pred = self.forward(X, w)
            return float(np.mean((y_pred - y) ** 2))

        best_w, best_fit = optimizer.optimize(fitness, verbose=verbose)
        self.weights = best_w
        self.train_mse = best_fit
        self.optimizer_history = optimizer.history
        self.trainer_ = "bat"
        return self
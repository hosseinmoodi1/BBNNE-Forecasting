"""
MLP expert with a single hidden layer (tau -> hidden -> 1).

Activation
----------
- Hidden layer : log-sigmoid   f(z) = 1 / (1 + exp(-z))
- Output layer : linear

Weight optimization is delegated to `BatAlgorithm`.
"""
from __future__ import annotations

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
        # FIX #4: optimizer_history is defined here so that it is always
        # accessible even if `fit()` is never called.
        self.weights: Optional[np.ndarray] = None
        self.train_mse: Optional[float] = None
        self.optimizer_history: Optional[Dict] = None

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
    ) -> "MLPExpert":
        """
        Train network weights with the Bat Algorithm.

        The fitness function is Mean Squared Error (MSE) on the provided
        training subset. MSE is the standard differentiable loss for
        regression and is compatible with the gradient-free search
        performed by the Bat Algorithm.
        """
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
        self.optimizer_history = optimizer.history  # exposed for plotting
        return self
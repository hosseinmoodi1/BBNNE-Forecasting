"""
Bat Algorithm (Yang, 2010).

Implements the standard Bat Algorithm with the following parameter
conventions from the BBNNE paper:

    f_i  = f_min + (f_max - f_min) * beta       (Eq. 4)
    v_i^t = v_i^{t-1} + (x_i - x_*) * f_i        (Eq. 5)
    x_i^t = x_i^{t-1} + v_i^t                    (Eq. 6, implicit)
    x_new = x_old + epsilon * A^t                (Eq. 7)
    A_i^{t+1} = alpha * A_i^t                    (Eq. 8)
    r_i^{t+1} = r_i^0 * (1 - exp(-gamma*t))      (Eq. 9)
    A_i -> 0, r_i -> r_i^0 as t -> inf           (Eq. 10)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class BatAlgorithm:
    """
    Bat Algorithm metaheuristic for continuous optimization.

    Parameters
    ----------
    n_params : int
        Dimension of the search space (number of MLP weights).
    n_pop : int
        Bat population size (N_p in the paper).
    max_iter : int
        Maximum number of iterations (I_max in the paper).
    f_min, f_max : float
        Minimum and maximum frequency bounds.
    alpha : float
        Loudness decay factor (in (0, 1)).
    gamma : float
        Pulse rate amplification factor (> 0).
    A0 : float
        Initial loudness.
    r0 : float
        Initial pulse emission rate (r_i^0 in the paper).
    bounds : tuple (lb, ub)
        Lower and upper bounds for every parameter.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        n_params: int,
        n_pop: int = 30,
        max_iter: int = 200,
        f_min: float = 0.0,
        f_max: float = 2.0,
        alpha: float = 0.9,
        gamma: float = 0.9,
        A0: float = 0.5,
        r0: float = 0.5,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        seed: Optional[int] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        if not (0.0 <= r0 <= 1.0):
            raise ValueError("r0 must be in [0, 1].")
        if not (0.0 <= A0):
            raise ValueError("A0 must be non-negative.")

        self.n_params = int(n_params)
        self.n_pop = int(n_pop)
        self.max_iter = int(max_iter)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.A0 = float(A0)
        self.r0 = float(r0)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.rng = np.random.default_rng(seed)

        # Diagnostic history
        self.history: Dict[str, List[float]] = {"best_fitness": []}

    # --------------------------------------------------------------------- #
    # Main optimization loop
    # --------------------------------------------------------------------- #
    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the Bat Algorithm to minimize `fitness_func`.

        Parameters
        ----------
        fitness_func : callable
            Maps a parameter vector (shape (n_params,)) to a scalar
            (lower is better).
        verbose : bool
            If True, print progress every 50 iterations.

        Returns
        -------
        best_position : np.ndarray
            Best solution found (shape (n_params,)).
        best_fitness : float
            Fitness value of the best solution.
        """
        lb, ub = self.bounds
        P = self.n_pop
        D = self.n_params

        # ---- Initialization ----
        positions = self.rng.uniform(lb, ub, size=(P, D))
        velocities = np.zeros((P, D), dtype=np.float64)
        loudness = np.full(P, self.A0, dtype=np.float64)   # A_i
        pulse_rate = np.full(P, self.r0, dtype=np.float64)  # r_i

        fitness = np.array([fitness_func(p) for p in positions], dtype=np.float64)
        best_idx = int(np.argmin(fitness))
        best_position = positions[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        self.history["best_fitness"].append(best_fitness)

        # ---- Main loop ----
        for t in range(1, self.max_iter + 1):
            mean_loudness = float(loudness.mean())

            for i in range(P):
                # Eq. 4: frequency
                beta = self.rng.random()
                freq = self.f_min + (self.f_max - self.f_min) * beta

                # Eq. 5 + 6: velocity and position update
                velocities[i] += (positions[i] - best_position) * freq
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Eq. 7: local random walk (with probability 1 - r_i)
                if self.rng.random() > pulse_rate[i]:
                    epsilon = self.rng.uniform(-1.0, 1.0, size=D)
                    candidate = best_position + epsilon * mean_loudness
                    candidate = np.clip(candidate, lb, ub)
                else:
                    candidate = positions[i].copy()

                candidate_fit = fitness_func(candidate)

                # Acceptance criterion: improvement + stochastic loudness
                if (candidate_fit < fitness[i]) and (self.rng.random() < loudness[i]):
                    positions[i] = candidate
                    fitness[i] = candidate_fit
                    # Eq. 8: loudness decreases
                    loudness[i] *= self.alpha
                    # Eq. 9: pulse rate increases toward r0
                    pulse_rate[i] = self.r0 * (1.0 - np.exp(-self.gamma * t))

                # Update global best
                if fitness[i] < best_fitness:
                    best_position = positions[i].copy()
                    best_fitness = float(fitness[i])

            self.history["best_fitness"].append(best_fitness)

            if verbose and (t % 50 == 0 or t == self.max_iter):
                print(f"  [BA] iter {t:3d}/{self.max_iter} | "
                      f"best fitness = {best_fitness:.6e}")

        return best_position, best_fitness
"""
Unit tests for BBNNE components.
Run with: pytest tests/
"""
import numpy as np
import pytest

from bbnne import BBNNE, BatAlgorithm, MLPExpert, rlsg
from bbnne.metrics import mape, mae, rmse, directional_accuracy, theils_u


# ---------- RLSG ---------- #
def test_rlsg_produces_distinct_lags():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 10))
    y = X[:, 3] + 0.1 * rng.normal(size=100)
    spaces, rs = rlsg(X, y, n_spaces=3, subspace_size=4, seed=0)
    assert len(spaces) == 3
    for s in spaces:
        assert len(s) == 4
        assert len(set(s)) == 4           # no duplicate lag inside subspace
    assert rs.shape == (10,)
    assert np.isclose(rs.sum(), 1.0)


# ---------- Bat Algorithm ---------- #
def test_bat_optimizes_quadratic():
    target = np.array([0.3, -0.5, 0.7])

    def fitness(w):
        return float(np.sum((w - target) ** 2))

    ba = BatAlgorithm(n_params=3, n_pop=20, max_iter=100,
                      bounds=(-2, 2), seed=1)
    best_w, best_fit = ba.optimize(fitness)
    assert best_fit < 0.05
    assert np.allclose(best_w, target, atol=0.3)


# ---------- MLP Expert ---------- #
def test_mlp_trains_on_simple_signal():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(80, 3))
    y = X[:, 0] + 0.5 * X[:, 1]
    expert = MLPExpert(n_input=3, n_hidden=6, seed=0)
    expert.fit(X, y, bat_params=dict(n_pop=10, max_iter=50))
    pred = expert.predict(X)
    assert pred.shape == (80,)
    assert np.mean((pred - y) ** 2) < 0.5


# ---------- BBNNE ---------- #
def test_bbnne_runs_end_to_end():
    rng = np.random.default_rng(0)
    # Synthetic random-walk-ish series
    prices = np.cumsum(rng.normal(scale=0.01, size=300))
    prices = (prices - prices.min()) / (prices.max() - prices.min()) * 2 - 1

    model = BBNNE(
        n_experts=3, max_lag=10, subspace_size=3, hidden_size=4,
        bat_params=dict(n_pop=6, max_iter=15),
        seed=0,
    )
    model.fit(prices, verbose=False)
    preds = model.predict(prices)
    assert preds.shape == (len(prices) - 10,)
    assert model.n_experts_fitted_ >= 1
    assert np.all(np.isfinite(model.expert_weights))


# ---------- Metrics ---------- #
def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 2.5, 2.0])
    assert mape(y, y) < 1e-9
    assert mae(y, y) < 1e-9
    assert rmse(y, y) < 1e-9
    assert directional_accuracy(y, y) == pytest.approx(100.0)
    assert theils_u(y, y) < 1e-9
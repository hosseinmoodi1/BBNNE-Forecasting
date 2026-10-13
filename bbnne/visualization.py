"""
Visualization utilities for BBNNE results.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_mape_vs_experts(
    results: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Reproduce Figure 1 of the paper: MAPE as a function of ensemble size.

    Parameters
    ----------
    results : dict
        {dataset_name: {n_experts: test_MAPE}}.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    n_datasets = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (name, curve) in zip(axes, results.items()):
        ks = sorted(curve.keys())
        vals = [curve[k] for k in ks]
        bars = ax.bar([str(k) for k in ks], vals, color="steelblue")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of experts")
        ax.set_ylabel("Test MAPE (%)")
        # Annotate values above each bar
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    for ax in axes[len(results):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "BBNNE prediction",
    save_path: Optional[str] = None,
) -> None:
    """Plot actual vs. predicted series on the test set."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual", linewidth=1.5)
    plt.plot(y_pred, label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_lag_spaces(lag_spaces: List[List[int]], top_n: int = 6) -> None:
    """Print the selected lag spaces in a table."""
    n_experts = min(len(lag_spaces), top_n)
    header = " | ".join(f"LS{i + 1}" for i in range(n_experts))
    print(header)
    print("-" * len(header))
    rows = [
        [f"lag{idx + 1}" for idx in lag_spaces[i][:n_experts]]
        for i in range(n_experts)
    ]
    for r in zip(*rows):
        print(" | ".join(r))
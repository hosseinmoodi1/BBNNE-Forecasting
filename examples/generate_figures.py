"""
generate_figures.py — Standalone figure generator for the BBNNE paper.

Reads existing results from the output directory and produces:

    fig1_mape_bars.png         — MAPE bar chart per dataset and model
    fig2_theilu_heatmap.png    — Theil's U heatmap
    fig3_predictions.png       — Actual vs. predicted series
    fig4_radar.png             — Normalized MAPE radar chart (log scale)
    fig5_wilcoxon_z.png        — |Z| bar chart for Wilcoxon tests

Inputs (must already exist in --output-dir):
    table6_summary.csv
    table7_wilcoxon_{SHPETRO,FOLAD,IBM,MSFT}.csv
    predictions.npz                (for fig3)

Usage
-----
    python examples/generate_figures.py --output-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # must precede pyplot
import matplotlib.pyplot as plt


# ===================================================================== #
# Configuration                                                         #
# ===================================================================== #
DATASET_ORDER: List[str] = ["SHPETRO", "FOLAD", "IBM", "MSFT"]
TABLES_MODELS: List[str] = ["BBNNE", "MLP", "RBF", "Bagging", "AdaBoost"]

COLORS: Dict[str, str] = {
    "BBNNE":    "#C0392B",
    "MLP":      "#2980B9",
    "RBF":      "#27AE60",
    "Bagging":  "#F39C12",
    "AdaBoost": "#8E44AD",
    "LSTM":     "#16A085",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ===================================================================== #
# Figure 1: MAPE bar chart                                              #
# ===================================================================== #
def fig1_mape_bars(summary: pd.DataFrame, fig_dir: Path) -> None:
    datasets = [
        ds for ds in DATASET_ORDER if ds in summary["Dataset"].unique()
    ]
    n_ds = len(datasets)
    fig, axes = plt.subplots(
        1, n_ds, figsize=(4 * n_ds, 4.2), sharey=False
    )
    if n_ds == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = summary[summary["Dataset"] == ds]
        sub = sub.set_index("Model").reindex(TABLES_MODELS).dropna(
            subset=["MAPE(%)"]
        )
        vals = sub["MAPE(%)"].values
        names = sub.index.tolist()
        colors = [COLORS.get(m, "gray") for m in names]
        bars = ax.bar(names, vals, color=colors,
                      edgecolor="black", linewidth=0.6)
        ax.set_title(ds, fontweight="bold")
        ax.set_ylabel("MAPE (%)" if ds == datasets[0] else "")
        ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        for b, m in zip(bars, names):
            if m == "BBNNE":
                b.set_edgecolor("black")
                b.set_linewidth(1.8)
    fig.suptitle("Test MAPE by dataset and model", fontsize=14, y=1.02)
    plt.tight_layout()
    out = fig_dir / "fig1_mape_bars.png"
    plt.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ===================================================================== #
# Figure 2: Theil's U heatmap                                           #
# ===================================================================== #
def fig2_theilu_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    pivot = summary.pivot_table(
        index="Model", columns="Dataset", values="TheilU"
    ).reindex(TABLES_MODELS)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Theil's U (lower is better)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="black", fontsize=10)
    ax.set_title("Theil's U statistic across models and datasets")
    plt.tight_layout()
    out = fig_dir / "fig2_theilu_heatmap.png"
    plt.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ===================================================================== #
# Figure 3: Actual vs. predicted                                        #
# ===================================================================== #
def fig3_predictions(results_dir: Path, fig_dir: Path) -> None:
    cache = results_dir / "predictions.npz"
    if not cache.exists():
        print(f"  ⚠ {cache.name} not found — skipping fig3")
        return
    data = np.load(cache)
    datasets = sorted(
        {k.split("__")[0] for k in data.files},
        key=lambda d: DATASET_ORDER.index(d)
        if d in DATASET_ORDER else 999,
    )
    models = ["BBNNE", "MLP"]
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(11, 2.5 * len(datasets))
    )
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        y_true = data[f"{ds}__y_true"]
        ax.plot(y_true, color="black", lw=1.5, label="Actual")
        for m in models:
            key = f"{ds}__{m}__y_pred"
            if key in data.files:
                ax.plot(data[key], lw=1.0, alpha=0.85,
                        color=COLORS.get(m, "gray"), label=m)
        ax.set_title(f"{ds} — Test predictions", fontweight="bold")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    out = fig_dir / "fig3_predictions.png"
    plt.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ===================================================================== #
# Figure 4: Radar chart (log scale)                                     #
# ===================================================================== #
def fig4_radar(summary: pd.DataFrame, fig_dir: Path) -> None:
    """
    Radar chart on a LOG scale.

    A linear-scale radar chart is dominated by the RBF model on the
    FOLAD axis (where RBF MAPE = 60.9%), which visually compresses all
    remaining models into the origin. Plotting log10 of the normalized
    MAPE avoids this distortion and makes the differences between the
    four best models visible, while still including RBF as a reference.
    """
    pivot = summary.pivot_table(
        index="Model", columns="Dataset", values="MAPE(%)"
    ).reindex(TABLES_MODELS)

    # Normalize so that the best model on each dataset is at 1.0
    best = pivot.min(axis=0)
    norm = pivot.divide(best, axis=1)

    # Log-transform to compress the dynamic range
    log_norm = np.log10(norm + 1e-3)

    labels = list(log_norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels),
                         endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           subplot_kw=dict(polar=True))
    for model in log_norm.index:
        vals = log_norm.loc[model].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, lw=1.8,
                color=COLORS.get(model, "gray"), label=model)
        ax.fill(angles, vals, alpha=0.06,
                color=COLORS.get(model, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(
        "Normalized MAPE (log scale; 0 = best per dataset)",
        fontweight="bold", pad=18,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05),
              fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig4_radar.png"
    plt.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ===================================================================== #
# Figure 5: Wilcoxon |Z| bar chart                                      #
# ===================================================================== #
def fig5_wilcoxon_z(results_dir: Path, fig_dir: Path) -> None:
    paths = [
        results_dir / f"table7_wilcoxon_{ds}.csv"
        for ds in DATASET_ORDER
        if (results_dir / f"table7_wilcoxon_{ds}.csv").exists()
    ]
    if not paths:
        print("  ⚠ no table7_wilcoxon files — skipping fig5")
        return
    rows = []
    for p in paths:
        ds = p.stem.replace("table7_wilcoxon_", "")
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows.append({
                "Dataset": ds, "Model": r["Model"],
                "AbsZ": abs(float(r["Z"])),
            })
    agg = pd.DataFrame(rows)
    pivot = agg.pivot_table(
        index="Model", columns="Dataset", values="AbsZ"
    )
    pivot = pivot.reindex(
        [m for m in TABLES_MODELS if m in pivot.index]
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(pivot.index))
    width = 0.18
    for i, ds in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[ds].values, width,
               label=ds, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("|Z| (higher = stronger superiority)")
    ax.set_title("Wilcoxon signed-rank |Z| — BBNNE vs. each baseline")
    ax.legend(title="Dataset", fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig5_wilcoxon_z.png"
    plt.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ===================================================================== #
# Main                                                                  #
# ===================================================================== #
def main():
    p = argparse.ArgumentParser(
        description="Generate BBNNE figures from existing results."
    )
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                   help="Directory containing results and where figures "
                        "will be written.")
    args = p.parse_args()

    results_dir = args.output_dir
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "table6_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Please run the pipeline first "
            f"(python examples/run_pipeline.py --stage experiments)."
        )
    summary = pd.read_csv(summary_path)

    print(f"Reading from: {results_dir}")
    print(f"Writing to:   {fig_dir}")
    print()

    fig1_mape_bars(summary, fig_dir)
    fig2_theilu_heatmap(summary, fig_dir)
    fig3_predictions(results_dir, fig_dir)
    fig4_radar(summary, fig_dir)
    fig5_wilcoxon_z(results_dir, fig_dir)

    print()
    print(f"All figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
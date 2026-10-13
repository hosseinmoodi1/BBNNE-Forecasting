"""
run_pipeline.py — Full BBNNE pipeline (Stage-6 settings).

Stage 6 configuration (validated via cascade experiments):
    shrinkage    = 0.5
    horizon      = 3
    feature_mode = "raw_lags"
    N_SEEDS      = 5

Note
----
The LSTM baseline is implemented in bbnne/benchmarks.py but is NOT
included in the main pipeline, consistent with the manuscript.
It is available as an optional extension for future work.

HORIZON-AWARE THEIL'S U
-----------------------
Theil's U is now computed with a horizon-matched denominator. Both the
model's error and the persistence denominator are evaluated over the
same h-step horizon, so the statistic is directly comparable to the
theoretical threshold U < 1.

Usage
-----
    python examples/run_pipeline.py --stage all
    python examples/run_pipeline.py --stage experiments
    python examples/run_pipeline.py --stage tables
    python examples/run_pipeline.py --stage figures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bbnne import BBNNE, evaluate
from bbnne.data_loader import load_tse_csv, load_us_csv, PAPER_WINDOWS
from bbnne.benchmarks import (
    fit_mlp, fit_rbf, fit_bagging, fit_adaboost,
)

# ===================================================================== #
# Configuration
# ===================================================================== #
BAT_PARAMS = dict(
    n_pop=30, max_iter=200,
    f_min=0.0, f_max=2.0,
    alpha=0.9, gamma=0.9,
    A0=0.5, r0=0.5,
    bounds=(-1.0, 1.0),
)

MAX_LAG = 15
SUBSPACE_SIZE = 5
N_EXPERTS = 7
HIDDEN_SIZE = 10

# --- Stage-6 settings (final) ---
N_BBNNE_SEEDS = 5
BBNNE_SHRINKAGE = 0.5
USE_BOOSTING = True
BBNNE_HORIZON = 3
BBNNE_FEATURE_MODE = "raw_lags"

# Benchmarks
BENCH_HIDDEN = 6
BENCH_BAGGING_N = 7
BENCH_ADABOOST_N = 5

TRAIN_RATIO = 0.8
SEED = 42

DATASET_ORDER: List[str] = ["SHPETRO", "FOLAD", "IBM", "MSFT"]
TABLES_MODELS: List[str] = ["BBNNE", "MLP", "RBF", "Bagging", "AdaBoost"]
MODELS_ORDER: List[str] = TABLES_MODELS  # kept for backward compatibility

COLORS: Dict[str, str] = {
    "BBNNE":    "#C0392B",
    "MLP":      "#2980B9",
    "RBF":      "#27AE60",
    "Bagging":  "#F39C12",
    "AdaBoost": "#8E44AD",
    "LSTM":     "#16A085",
}

# ===================================================================== #
# Utilities
# ===================================================================== #
def _default_results_dir() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "pyproject.toml").exists():
            return p / "results"
        p = p.parent
    return Path.cwd() / "results"

def get_datasets(data_dir: Path):
    return {
        "SHPETRO": (data_dir / "SHPETRO.csv", load_tse_csv),
        "FOLAD":   (data_dir / "FEOLAD.csv",  load_tse_csv),
        "IBM":     (
            data_dir / "DailyData_#IBM_2026-09-14.csv", load_us_csv
        ),
        "MSFT":    (
            data_dir / "DailyData_#MSFT_2026-09-14.csv", load_us_csv
        ),
    }

# ===================================================================== #
# Model runners
# ===================================================================== #
def run_bbnne(prices_train: np.ndarray,
              prices_test: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(N_BBNNE_SEEDS):
        seed_i = SEED + i * 13
        m = BBNNE(
            n_experts=N_EXPERTS,
            max_lag=MAX_LAG,
            subspace_size=SUBSPACE_SIZE,
            hidden_size=HIDDEN_SIZE,
            bat_params=BAT_PARAMS,
            seed=seed_i,
            sampling_seed=seed_i,
            residual_shrinkage=BBNNE_SHRINKAGE,
            use_boosting=USE_BOOSTING,
            horizon=BBNNE_HORIZON,
            feature_mode=BBNNE_FEATURE_MODE,
        )
        m.fit(prices_train, verbose=False)
        preds.append(m.predict(prices_test))
    return np.mean(preds, axis=0)

def run_mlp(tr, te):
    return fit_mlp(
        tr, te, max_lag=MAX_LAG, hidden=BENCH_HIDDEN,
        seed=SEED, shrinkage=BBNNE_SHRINKAGE,
        horizon=BBNNE_HORIZON, feature_mode=BBNNE_FEATURE_MODE,
    )

def run_rbf(tr, te):
    return fit_rbf(
        tr, te, max_lag=MAX_LAG, n_centers=BENCH_HIDDEN,
        seed=SEED, shrinkage=BBNNE_SHRINKAGE,
        horizon=BBNNE_HORIZON, feature_mode=BBNNE_FEATURE_MODE,
    )

def run_bagging(tr, te):
    return fit_bagging(
        tr, te, max_lag=MAX_LAG,
        n_estimators=BENCH_BAGGING_N,
        hidden=BENCH_HIDDEN, seed=SEED,
        shrinkage=BBNNE_SHRINKAGE,
        horizon=BBNNE_HORIZON, feature_mode=BBNNE_FEATURE_MODE,
    )

def run_adaboost(tr, te):
    return fit_adaboost(
        tr, te, max_lag=MAX_LAG,
        n_estimators=BENCH_ADABOOST_N,
        hidden=BENCH_HIDDEN, seed=SEED,
        shrinkage=BBNNE_SHRINKAGE,
        horizon=BBNNE_HORIZON, feature_mode=BBNNE_FEATURE_MODE,
    )

MODEL_RUNNERS = {
    "BBNNE":    run_bbnne,
    "MLP":      run_mlp,
    "RBF":      run_rbf,
    "Bagging":  run_bagging,
    "AdaBoost": run_adaboost,
}

# ===================================================================== #
# Wilcoxon signed-rank test
# ===================================================================== #
def _wilcoxon_z(proposed_err: np.ndarray,
                benchmark_err: np.ndarray
                ) -> Tuple[float, float, float]:
    """Signed Z; positive => benchmark has larger errors (BBNNE better)."""
    proposed_err = np.asarray(proposed_err, dtype=np.float64)
    benchmark_err = np.asarray(benchmark_err, dtype=np.float64)

    d = benchmark_err - proposed_err
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    abs_d = np.abs(d)
    ranks = rankdata(abs_d)

    _, counts = np.unique(abs_d, return_counts=True)
    tie_corr = float(np.sum(counts ** 3 - counts))

    mu = n * (n + 1) / 4.0
    sigma_sq = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
    if sigma_sq <= 0:
        return float("nan"), float("nan"), float("nan")
    sigma = float(np.sqrt(sigma_sq))

    w_plus = float(np.sum(ranks[d > 0]))
    diff = w_plus - mu
    correction = 0.5 * np.sign(diff) if diff != 0 else 0.0
    Z = (diff - correction) / sigma
    p = 2.0 * (1.0 - norm.cdf(abs(Z)))
    return w_plus, float(Z), float(p)

def wilcoxon_comparison(results: Dict,
                        reference: str = "BBNNE") -> pd.DataFrame:
    if reference not in results:
        return pd.DataFrame()
    ref_err = results[reference]["abs_err"]
    rows = []
    for name, res in results.items():
        if name == reference or name.startswith("_"):
            continue
        bench_err = res["abs_err"]
        if len(bench_err) != len(ref_err):
            continue
        try:
            W, Z, p = _wilcoxon_z(ref_err, bench_err)
        except Exception:
            W, Z, p = np.nan, np.nan, np.nan
        rows.append({
            "Model": name,
            "W": W,
            "Z": Z,
            "|Z|": abs(Z) if np.isfinite(Z) else np.nan,
            "p-value": p,
            "significant": "yes" if (np.isfinite(p) and p < 0.05) else "no",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("Model").reindex(
            [m for m in MODELS_ORDER if m in df["Model"].values]
        ).reset_index()
    return df

# ===================================================================== #
# Experiments (HORIZON-AWARE)
# ===================================================================== #
def run_experiment(dataset_name, filepath, loader, verbose=True):
    if verbose:
        print("=" * 78)
        print(f"  Dataset: {dataset_name}  |  "
              f"h={BBNNE_HORIZON}  mode={BBNNE_FEATURE_MODE}  "
              f"gamma={BBNNE_SHRINKAGE}  N_SEEDS={N_BBNNE_SEEDS}")
        print("=" * 78)

    window = PAPER_WINDOWS.get(dataset_name)
    try:
        prices = loader(filepath, date_window=window)
    except TypeError:
        prices = loader(filepath)

    split = int(len(prices) * TRAIN_RATIO)
    train_raw = prices[:split]
    test_raw = prices[split - MAX_LAG:]
    y_true_full = prices[split:]
    n_test_full = len(y_true_full)

    h = BBNNE_HORIZON

    # Number of predictions produced by every runner:
    #   len(test_raw) - MAX_LAG - h + 1
    n_pred = len(test_raw) - MAX_LAG - h + 1

    # Correct alignment: y_pred[j] ~ prices[split + h - 1 + j]
    y_true_aligned = y_true_full[h - 1 : h - 1 + n_pred]

    # Persistence (h-step ahead): prediction for the target at index
    # (split + h - 1 + j) is the price at index (split - 1 + j).
    persistence_pred = np.concatenate([
        [prices[split - 1]],
        y_true_full[: n_pred - 1],
    ])
    baseline_metrics = evaluate(
        y_true_aligned, persistence_pred, horizon=h
    )

    if verbose:
        print(f"  Total: {len(prices)} | "
              f"Train: {len(train_raw)} | Test: {n_test_full} | "
              f"Predictions: {n_pred}")
        print(f"  Persistence (h={h}): "
              f"MAPE={baseline_metrics['MAPE(%)']:.4f}% | "
              f"TheilU={baseline_metrics['TheilU']:.4f}")

    results: Dict[str, dict] = {}
    for model_name, runner in MODEL_RUNNERS.items():
        if verbose:
            print(f"\n  -- {model_name} --")
        t0 = time.time()
        try:
            y_pred = runner(train_raw, test_raw)
        except Exception as exc:
            if verbose:
                print(f"     [SKIP] {exc}")
            continue
        if y_pred is None:
            if verbose:
                print(f"     [SKIP] runner returned None")
            continue

        elapsed = time.time() - t0

        if len(y_pred) > n_pred:
            y_pred = y_pred[:n_pred]
        n = len(y_pred)
        y_t = y_true_aligned[:n]

        # Horizon-aware Theil's U is applied here
        m = evaluate(y_t, y_pred, horizon=h)
        m["time_sec"] = round(elapsed, 2)
        results[model_name] = {
            "metrics": m,
            "y_true": y_t,
            "y_pred": y_pred,
            "abs_err": np.abs(y_t - y_pred),
        }

        if verbose:
            print(f"     MAPE={m['MAPE(%)']:.4f}% | MAE={m['MAE']:.4f} | "
                  f"RMSE={m['RMSE']:.4f} | DA={m['DA(%)']:.2f}% | "
                  f"TheilU={m['TheilU']:.4f} | R2={m['R2']:.4f} | "
                  f"time={elapsed:.1f}s")

    results["_persistence"] = {"metrics": baseline_metrics}
    return results

def build_summary_table(all_results: Dict) -> pd.DataFrame:
    rows = []
    for dataset, res in all_results.items():
        for model, payload in res.items():
            if model.startswith("_"):
                continue
            m = payload["metrics"]
            rows.append({
                "Dataset": dataset, "Model": model,
                "MAPE(%)": m["MAPE(%)"], "MAE": m["MAE"],
                "RMSE": m["RMSE"], "DA(%)": m["DA(%)"],
                "TheilU": m["TheilU"], "R2": m.get("R2", np.nan),
                "Time(s)": m["time_sec"],
            })
    df = pd.DataFrame(rows)
    df["Dataset"] = pd.Categorical(
        df["Dataset"], categories=DATASET_ORDER, ordered=True
    )
    return df.sort_values(["Dataset", "Model"]).reset_index(drop=True).round(4)

def _save_predictions(all_results: Dict, output_dir: Path) -> None:
    arrays = {}
    for ds, res in all_results.items():
        y_saved = False
        for model, payload in res.items():
            if model.startswith("_"):
                continue
            if not y_saved:
                arrays[f"{ds}__y_true"] = payload["y_true"]
                y_saved = True
            arrays[f"{ds}__{model}__y_pred"] = payload["y_pred"]
    np.savez(output_dir / "predictions.npz", **arrays)

def stage_experiments(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = get_datasets(args.data_dir)
    verbose = not args.quiet

    all_results: Dict = {}
    wilcoxon_tables: Dict[str, pd.DataFrame] = {}

    for name in DATASET_ORDER:
        if name not in datasets:
            continue
        fp, loader = datasets[name]
        if not fp.exists():
            print(f"[SKIP] {name}: file not found at {fp}")
            continue
        print(f"\n>>> Running {name} ...")
        try:
            res = run_experiment(name, fp, loader, verbose=verbose)
            all_results[name] = res
            wilcoxon_tables[name] = wilcoxon_comparison(res, "BBNNE")
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No experiments completed.")
        return

    summary = build_summary_table(all_results)
    summary.to_csv(args.output_dir / "table6_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("  TABLE 6 — Test errors for all ML models")
    print("=" * 78)
    print(summary.to_string(index=False))

    print("\n  WINNER per dataset (lowest MAPE):")
    for ds in summary["Dataset"].unique():
        sub = summary[summary["Dataset"] == ds]
        best = sub.loc[sub["MAPE(%)"].idxmin()]
        print(f"    {ds:10s} -> {best['Model']:12s} "
              f"(MAPE={best['MAPE(%)']:.4f}%, "
              f"DA={best['DA(%)']:.2f}%, "
              f"TheilU={best['TheilU']:.4f})")

    for name, tbl in wilcoxon_tables.items():
        if tbl.empty:
            continue
        path = args.output_dir / f"table7_wilcoxon_{name}.csv"
        tbl.to_csv(path, index=False)

    _save_predictions(all_results, args.output_dir)

    serializable = {
        ds: {mdl: {"metrics": p["metrics"]}
             for mdl, p in res.items() if not mdl.startswith("_")}
        for ds, res in all_results.items()
    }
    with open(args.output_dir / "all_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  Saved: {args.output_dir}")

# ===================================================================== #
# Stage 2: LaTeX tables
# ===================================================================== #
def _fmt(v, digits=3):
    if v is None:
        return "--"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "--"
    return f"{fv:.{digits}f}"

def build_table6(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Test error rates for all datasets and models "
        r"(horizon $h=3$, raw-lag inputs). Lower is better for MAE, "
        r"MAPE, RMSE, and Theil's $U$; higher is better for DA.}",
        r"\label{tab:results}",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Model & MAE & MAPE(\%) & RMSE & DA(\%) & TheilU \\",
        r"\midrule",
    ]
    datasets = [ds for ds in DATASET_ORDER
                if ds in summary["Dataset"].unique()]
    for ds_idx, ds in enumerate(datasets):
        sub = summary[summary["Dataset"] == ds]
        sub = sub.set_index("Model").reindex(TABLES_MODELS).dropna(
            how="all"
        )
        for m_idx, (model, row) in enumerate(sub.iterrows()):
            ds_cell = ds if m_idx == 0 else ""
            model_cell = (r"\textbf{" + model + r"}"
                          if model == "BBNNE" else model)
            cells = [
                _fmt(row["MAE"]), _fmt(row["MAPE(%)"]),
                _fmt(row["RMSE"]), _fmt(row["DA(%)"]),
                _fmt(row["TheilU"]),
            ]
            if model == "BBNNE":
                cells = [r"\textbf{" + c + r"}" for c in cells]
            lines.append(
                f"{ds_cell} & {model_cell} & " + " & ".join(cells)
                + r" \\"
            )
        if ds_idx < len(datasets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)

def build_table7(results_dir: Path) -> str:
    paths = [results_dir / f"table7_wilcoxon_{ds}.csv"
             for ds in DATASET_ORDER
             if (results_dir / f"table7_wilcoxon_{ds}.csv").exists()]
    if not paths:
        return "% No Wilcoxon tables found."
    rows = []
    for p in paths:
        ds = p.stem.replace("table7_wilcoxon_", "")
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows.append({
                "Dataset": ds, "Model": r["Model"],
                "Z": float(r["Z"]),
                "p": float(r["p-value"]),
                "sig": r["significant"],
            })
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Wilcoxon signed-rank test between BBNNE and each "
        r"benchmark on out-of-sample absolute errors. A \emph{positive} "
        r"$Z$ indicates the benchmark has systematically larger errors "
        r"than BBNNE. $p < 0.001$ is reported as $<0.001$.}",
        r"\label{tab:wilcoxon}",
        r"\small",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Dataset & Benchmark & $Z$ & $p$-value & Significant \\",
        r"\midrule",
    ]
    for ds_idx, ds in enumerate(DATASET_ORDER):
        sub = [r for r in rows if r["Dataset"] == ds]
        if not sub:
            continue
        sub.sort(key=lambda x: TABLES_MODELS.index(x["Model"])
                 if x["Model"] in TABLES_MODELS else 999)
        for m_idx, r in enumerate(sub):
            ds_cell = ds if m_idx == 0 else ""
            z_str = f"{r['Z']:.3f}"
            p_str = r"$<0.001$" if r["p"] < 0.001 else f"{r['p']:.3f}"
            sig_str = r"\checkmark" if r["sig"] == "yes" else "--"
            lines.append(
                f"{ds_cell} & {r['Model']} & {z_str} & {p_str} & "
                f"{sig_str} \\\\"
            )
        if ds_idx < len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)

def stage_tables(args):
    summary_path = args.output_dir / "table6_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run --stage experiments first."
        )
    summary = pd.read_csv(summary_path)
    out_dir = args.output_dir / "latex"
    out_dir.mkdir(exist_ok=True)

    t6 = build_table6(summary)
    t7 = build_table7(args.output_dir)

    (out_dir / "table6.tex").write_text(t6, encoding="utf-8")
    (out_dir / "table7.tex").write_text(t7, encoding="utf-8")
    combined = (
        "% Auto-generated LaTeX tables for the BBNNE paper.\n"
        "% Requires: \\usepackage{booktabs, amssymb}\n\n"
        + t6 + "\n\n" + t7 + "\n"
    )
    (out_dir / "tables_combined.tex").write_text(
        combined, encoding="utf-8"
    )
    print(f"  Saved: {out_dir / 'table6.tex'}")
    print(f"  Saved: {out_dir / 'table7.tex'}")
    print(f"  Saved: {out_dir / 'tables_combined.tex'}")

# ===================================================================== #
# Stage 3: Figures
# ===================================================================== #
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

def _load_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "table6_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    return pd.read_csv(path)

def fig1_mape_bars(summary, fig_dir):
    datasets = [ds for ds in DATASET_ORDER
                if ds in summary["Dataset"].unique()]
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 4.2),
                             sharey=False)
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
    fig.suptitle("Test MAPE by dataset and model (h=3)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = fig_dir / "fig1_mape_bars.png"
    plt.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")

def fig2_theilu_heatmap(summary, fig_dir):
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
    ax.set_title("Theil's U across models and datasets")
    plt.tight_layout()
    out = fig_dir / "fig2_theilu_heatmap.png"
    plt.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")

def fig3_predictions(results_dir, fig_dir):
    cache = results_dir / "predictions.npz"
    if not cache.exists():
        print(f"  Skip fig3: {cache.name} not found")
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
        ax.set_title(f"{ds} — Test predictions (h=3)",
                     fontweight="bold")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    out = fig_dir / "fig3_predictions.png"
    plt.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")

def fig4_radar(summary, fig_dir):
    pivot = summary.pivot_table(
        index="Model", columns="Dataset", values="MAPE(%)"
    ).reindex(TABLES_MODELS)
    best = pivot.min(axis=0)
    norm = pivot.divide(best, axis=1)
    labels = list(norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels),
                         endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))
    for model in norm.index:
        vals = norm.loc[model].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, lw=1.6,
                color=COLORS.get(model, "gray"), label=model)
        ax.fill(angles, vals, alpha=0.08,
                color=COLORS.get(model, "gray"))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Normalized MAPE (1.0 = best per dataset)",
                 fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05),
              fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig4_radar.png"
    plt.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")

def fig5_wilcoxon_z(results_dir, fig_dir):
    paths = [results_dir / f"table7_wilcoxon_{ds}.csv"
             for ds in DATASET_ORDER
             if (results_dir / f"table7_wilcoxon_{ds}.csv").exists()]
    if not paths:
        print("  Skip fig5: no wilcoxon files")
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
    ax.set_title("Wilcoxon signed-rank |Z| — BBNNE vs each baseline")
    ax.legend(title="Dataset", fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig5_wilcoxon_z.png"
    plt.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")

def stage_figures(args):
    summary = _load_summary(args.output_dir)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    print("Generating figures...")
    fig1_mape_bars(summary, fig_dir)
    fig2_theilu_heatmap(summary, fig_dir)
    fig3_predictions(args.output_dir, fig_dir)
    fig4_radar(summary, fig_dir)
    fig5_wilcoxon_z(args.output_dir, fig_dir)
    print(f"\n  All figures saved to: {fig_dir}")

# ===================================================================== #
# CLI
# ===================================================================== #
def parse_args():
    p = argparse.ArgumentParser(
        description="BBNNE pipeline (Stage-6: shrinkage=0.5, "
                    "horizon=3, raw_lags, N_SEEDS=5)."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path,
                   default=_default_results_dir())
    p.add_argument("--stage",
                   choices=["all", "experiments", "tables", "figures"],
                   default="all")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        print("\n" + "#" * 78)
        print("#  Stage 1/3 — Main experiments")
        print("#" * 78)
        stage_experiments(args)

        print("\n" + "#" * 78)
        print("#  Stage 2/3 — LaTeX tables")
        print("#" * 78)
        stage_tables(args)

        print("\n" + "#" * 78)
        print("#  Stage 3/3 — Figures")
        print("#" * 78)
        stage_figures(args)

        print("\n" + "=" * 78)
        print("  Pipeline complete. Outputs saved to:")
        print(f"    {args.output_dir}")
        print(f"    {args.output_dir / 'latex'}")
        print(f"    {args.output_dir / 'figures'}")
        print("=" * 78)

    elif args.stage == "experiments":
        stage_experiments(args)
    elif args.stage == "tables":
        stage_tables(args)
    elif args.stage == "figures":
        stage_figures(args)

if __name__ == "__main__":
    main()
"""
run_ablation.py — Ablation study for BBNNE (Stage-6 settings).

Variants
--------
Full BBNNE         : RLSG subspaces + Boosting
BBNNE w/o RLSG     : full feature matrix + Boosting
BBNNE w/o Boosting : RLSG subspaces + uniform averaging

Stage-6 configuration:
    shrinkage    = 0.5
    horizon      = 3
    feature_mode = "raw_lags"
    N_SEEDS      = 5

HORIZON-AWARE THEIL'S U
-----------------------
Theil's U is computed with a horizon-matched denominator. See
bbnne.metrics.theils_u for details.

Usage
-----
    python examples/run_ablation.py
    python examples/run_ablation.py --dataset IBM
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
from scipy.stats import rankdata, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bbnne.ablation import BBNNEAblation
from bbnne.metrics import evaluate
from bbnne.data_loader import load_tse_csv, load_us_csv, PAPER_WINDOWS


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

N_SEEDS = 5
SHRINKAGE = 0.5
HORIZON = 3
FEATURE_MODE = "raw_lags"

TRAIN_RATIO = 0.8
SEED = 42

DATASET_ORDER: List[str] = ["SHPETRO", "FOLAD", "IBM", "MSFT"]

VARIANTS: Dict[str, str] = {
    "Full BBNNE":         "full",
    "BBNNE w/o RLSG":     "wo_rlsg",
    "BBNNE w/o Boosting": "wo_boosting",
}

COLORS: Dict[str, str] = {
    "Full BBNNE":         "#C0392B",
    "BBNNE w/o RLSG":     "#2980B9",
    "BBNNE w/o Boosting": "#F39C12",
}


# ===================================================================== #
# Helpers
# ===================================================================== #
def _default_results_dir() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "pyproject.toml").exists():
            return p / "results"
        p = p.parent
    return Path.cwd() / "results"


def get_datasets(data_dir: Path) -> Dict[str, Tuple[Path, callable]]:
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
# Model runner
# ===================================================================== #
def run_variant(variant_code, train_prices, test_prices):
    preds = []
    for i in range(N_SEEDS):
        seed_i = SEED + i * 13
        model = BBNNEAblation(
            n_experts=N_EXPERTS,
            max_lag=MAX_LAG,
            subspace_size=SUBSPACE_SIZE,
            hidden_size=HIDDEN_SIZE,
            bat_params=BAT_PARAMS,
            seed=seed_i,
            sampling_seed=seed_i,
            variant=variant_code,
            residual_shrinkage=SHRINKAGE,
            horizon=HORIZON,
            feature_mode=FEATURE_MODE,
        )
        model.fit(train_prices, verbose=False)
        preds.append(model.predict(test_prices))
    return np.mean(preds, axis=0)


# ===================================================================== #
# Experiment
# ===================================================================== #
def run_dataset(dataset_name, filepath, loader, verbose=True):
    if verbose:
        print("=" * 78)
        print(f"  Ablation — Dataset: {dataset_name}  |  "
              f"h={HORIZON}  mode={FEATURE_MODE}  "
              f"gamma={SHRINKAGE}  N_SEEDS={N_SEEDS}")
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

    h = HORIZON
    n_pred = len(test_raw) - MAX_LAG - h + 1
    y_true_aligned = y_true_full[h - 1 : h - 1 + n_pred]

    if verbose:
        print(f"  Samples: train={len(train_raw)} | "
              f"test={len(y_true_full)} | predictions={n_pred}")

    results: Dict[str, dict] = {}

    for variant_name, variant_code in VARIANTS.items():
        if verbose:
            print(f"\n  -- {variant_name} --")
        t0 = time.time()
        try:
            y_pred_all = run_variant(
                variant_code, train_raw, test_raw
            )
        except Exception as exc:
            if verbose:
                print(f"     [SKIP] {exc}")
                import traceback
                traceback.print_exc()
            continue

        elapsed = time.time() - t0

        if len(y_pred_all) > n_pred:
            y_pred_all = y_pred_all[:n_pred]
        n = len(y_pred_all)
        y_pred = y_pred_all[:n]
        y_t = y_true_aligned[:n]

        # Horizon-aware Theil's U
        metrics = evaluate(y_t, y_pred, horizon=h)
        metrics["time_sec"] = round(elapsed, 2)

        results[variant_name] = {
            "metrics": metrics,
            "y_true": y_t,
            "y_pred": y_pred,
            "abs_err": np.abs(y_t - y_pred),
        }

        if verbose:
            print(
                f"     MAPE={metrics['MAPE(%)']:.4f}% | "
                f"MAE={metrics['MAE']:.4f} | "
                f"RMSE={metrics['RMSE']:.4f} | "
                f"DA={metrics['DA(%)']:.2f}% | "
                f"TheilU={metrics['TheilU']:.4f} | "
                f"R2={metrics['R2']:.4f} | "
                f"time={elapsed:.1f}s"
            )

    return results


# ===================================================================== #
# Summary tables
# ===================================================================== #
def build_ablation_table(all_results: Dict) -> pd.DataFrame:
    rows = []
    for ds in DATASET_ORDER:
        if ds not in all_results:
            continue
        for variant in VARIANTS:
            if variant not in all_results[ds]:
                continue
            m = all_results[ds][variant]["metrics"]
            rows.append({
                "Dataset": ds, "Variant": variant,
                "MAPE(%)": m["MAPE(%)"], "MAE": m["MAE"],
                "RMSE": m["RMSE"], "DA(%)": m["DA(%)"],
                "TheilU": m["TheilU"], "R2": m["R2"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Dataset"] = pd.Categorical(
            df["Dataset"], categories=DATASET_ORDER, ordered=True
        )
        df["Variant"] = pd.Categorical(
            df["Variant"], categories=list(VARIANTS.keys()), ordered=True
        )
        df = df.sort_values(["Dataset", "Variant"]).reset_index(drop=True)
    return df.round(4)


def paired_wilcoxon_full_vs_variant(all_results: Dict) -> pd.DataFrame:
    """Paired Wilcoxon: Full BBNNE vs. each variant."""
    rows = []
    for ds in DATASET_ORDER:
        if ds not in all_results:
            continue
        if "Full BBNNE" not in all_results[ds]:
            continue
        ref_err = all_results[ds]["Full BBNNE"]["abs_err"]

        for variant in VARIANTS:
            if variant == "Full BBNNE":
                continue
            if variant not in all_results[ds]:
                continue
            var_err = all_results[ds][variant]["abs_err"]
            if len(var_err) != len(ref_err):
                continue

            d = var_err - ref_err
            d = d[d != 0.0]
            n = len(d)
            if n == 0:
                rows.append({
                    "Dataset": ds, "Variant": variant,
                    "Z": np.nan, "p-value": np.nan,
                    "Full better?": "--",
                })
                continue

            try:
                _, p = wilcoxon(ref_err, var_err, zero_method="wilcox")
                abs_d = np.abs(d)
                ranks = rankdata(abs_d)
                _, counts = np.unique(abs_d, return_counts=True)
                tie_corr = float(np.sum(counts ** 3 - counts))
                mu = n * (n + 1) / 4.0
                sigma_sq = (
                    n * (n + 1) * (2 * n + 1) / 24.0
                    - tie_corr / 48.0
                )
                if sigma_sq <= 0:
                    Z = np.nan
                else:
                    w_plus = float(np.sum(ranks[d > 0]))
                    diff = w_plus - mu
                    corr = 0.5 * np.sign(diff) if diff != 0 else 0.0
                    Z = (diff - corr) / np.sqrt(sigma_sq)

                if p < 0.05 and ref_err.mean() < var_err.mean():
                    decision = "yes"
                elif p < 0.05 and ref_err.mean() > var_err.mean():
                    decision = "no"
                else:
                    decision = "tie/n.s."

                rows.append({
                    "Dataset": ds, "Variant": variant,
                    "Z": Z, "p-value": p,
                    "Full better?": decision,
                })
            except Exception:
                rows.append({
                    "Dataset": ds, "Variant": variant,
                    "Z": np.nan, "p-value": np.nan,
                    "Full better?": "--",
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Dataset"] = pd.Categorical(
            df["Dataset"], categories=DATASET_ORDER, ordered=True
        )
        df["Variant"] = pd.Categorical(
            df["Variant"], categories=list(VARIANTS.keys()), ordered=True
        )
        df = df.sort_values(["Dataset", "Variant"]).reset_index(drop=True)
    return df.round(4)


# ===================================================================== #
# LaTeX table
# ===================================================================== #
def _fmt(v, digits: int = 3) -> str:
    try:
        fv = float(v)
        return f"{fv:.{digits}f}" if np.isfinite(fv) else "--"
    except Exception:
        return "--"


def build_latex_ablation_table(df: pd.DataFrame,
                               wtable: pd.DataFrame) -> str:
    caption = (
        r"Ablation study on the contribution of each component in the "
        r"BBNNE framework (horizon $h=3$, raw-lag inputs). "
        r"``Full BBNNE'' uses both RLSG feature subspaces and "
        r"error-driven instance re-weighting (boosting). "
        r"Theil's $U$ is computed with a horizon-matched denominator. "
        r"The column ``Full better?'' reports the paired Wilcoxon "
        r"signed-rank test between Full BBNNE and each variant "
        r"(positive $Z$ indicates the variant has systematically "
        r"larger errors); ``n.s.'' stands for not significant at "
        r"$p = 0.05$."
    )

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{llccccc l}",
        r"\toprule",
        r"Dataset & Variant & MAE & MAPE(\%) & RMSE & DA(\%) & "
        r"TheilU & Full better? \\",
        r"\midrule",
    ]

    wilcox_lookup: Dict[Tuple[str, str], str] = {}
    if not wtable.empty:
        for _, r in wtable.iterrows():
            key = (str(r["Dataset"]), str(r["Variant"]))
            wilcox_lookup[key] = str(r["Full better?"])

    for ds_idx, ds in enumerate(DATASET_ORDER):
        sub = df[df["Dataset"] == ds]
        if sub.empty:
            continue
        for v_idx, (_, row) in enumerate(sub.iterrows()):
            ds_cell = ds if v_idx == 0 else ""
            v_name = row["Variant"]
            v_cell = (
                r"\textbf{" + v_name + r"}"
                if v_name == "Full BBNNE" else v_name
            )
            cells = [
                _fmt(row["MAE"]), _fmt(row["MAPE(%)"]),
                _fmt(row["RMSE"]), _fmt(row["DA(%)"]),
                _fmt(row["TheilU"]),
            ]
            if v_name == "Full BBNNE":
                cells = [r"\textbf{" + c + r"}" for c in cells]

            if v_name == "Full BBNNE":
                sig_cell = "--"
            else:
                sig_cell = wilcox_lookup.get((ds, v_name), "--")
                if sig_cell == "yes":
                    sig_cell = r"\checkmark"
                elif sig_cell == "no":
                    sig_cell = r"$\times$"
                elif sig_cell == "tie/n.s.":
                    sig_cell = r"n.s."

            lines.append(
                f"{ds_cell} & {v_cell} & " + " & ".join(cells)
                + f" & {sig_cell} \\\\"
            )
        if ds_idx < len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ===================================================================== #
# Figure: boxplot of absolute errors
# ===================================================================== #
def plot_ablation_box(all_results: Dict, fig_dir: Path) -> None:
    datasets = [ds for ds in DATASET_ORDER if ds in all_results]
    if not datasets:
        return

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.2 * len(datasets), 4.2),
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        data, labels, colors = [], [], []
        for v in VARIANTS:
            if v in all_results[ds]:
                data.append(all_results[ds][v]["abs_err"])
                labels.append(v.replace(" ", "\n"))
                colors.append(COLORS.get(v, "gray"))
        if not data:
            continue
        bp = ax.boxplot(
            data, patch_artist=True, showfliers=False, widths=0.6,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ds, fontweight="bold")
        if ds == datasets[0]:
            ax.set_ylabel("Absolute error (price units)")

    plt.tight_layout()
    out = fig_dir / "fig_ablation_box.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ===================================================================== #
# Prediction caching
# ===================================================================== #
def save_predictions(all_results: Dict, output_dir: Path) -> None:
    arrays = {}
    for ds, variants in all_results.items():
        y_saved = False
        for name, payload in variants.items():
            if not y_saved:
                arrays[f"{ds}__y_true"] = payload["y_true"]
                y_saved = True
            arrays[f"{ds}__{name}__y_pred"] = payload["y_pred"]
    np.savez(output_dir / "ablation_predictions.npz", **arrays)


# ===================================================================== #
# CLI
# ===================================================================== #
def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation study for BBNNE (Stage-6 settings)."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path,
                   default=_default_results_dir())
    p.add_argument("--dataset", choices=DATASET_ORDER, default=None,
                   help="Run only on this dataset (default: all).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    print(">>> Ablation study (Stage-6 settings) starting <<<")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latex").mkdir(exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    verbose = not args.quiet

    print(f">>> Data dir:   {args.data_dir.resolve()}")
    print(f">>> Output dir: {args.output_dir.resolve()}")

    datasets = get_datasets(args.data_dir)
    all_results: Dict[str, Dict[str, dict]] = {}

    targets = [args.dataset] if args.dataset else DATASET_ORDER
    for name in targets:
        if name not in datasets:
            continue
        fp, loader = datasets[name]
        if not fp.exists():
            print(f"[SKIP] {name}: file not found at {fp}")
            continue
        print(f"\n>>> Running {name} ...")
        try:
            all_results[name] = run_dataset(
                name, fp, loader, verbose=verbose
            )
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNo experiments completed. Check the data files.")
        return

    df = build_ablation_table(all_results)
    csv_path = args.output_dir / "table_ablation.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 78)
    print("  ABLATION TABLE")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\n  Saved: {csv_path}")

    wtable = paired_wilcoxon_full_vs_variant(all_results)
    if not wtable.empty:
        wpath = args.output_dir / "table_ablation_wilcoxon.csv"
        wtable.to_csv(wpath, index=False)
        print("\n  WILCOXON: Full BBNNE vs. each variant")
        print(wtable.to_string(index=False))
        print(f"\n  Saved: {wpath}")

    tex = build_latex_ablation_table(df, wtable)
    tex_path = args.output_dir / "latex" / "table_ablation.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  Saved: {tex_path}")

    serializable = {
        ds: {var: {"metrics": payload["metrics"]}
             for var, payload in variants.items()}
        for ds, variants in all_results.items()
    }
    with open(args.output_dir / "ablation_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {args.output_dir / 'ablation_metrics.json'}")

    save_predictions(all_results, args.output_dir)
    print(f"  Saved: {args.output_dir / 'ablation_predictions.npz'}")

    plot_ablation_box(all_results, args.output_dir / "figures")

    print("\n>>> Ablation study complete <<<")


if __name__ == "__main__":
    main()
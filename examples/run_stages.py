"""
run_stages.py — Multi-stage improvement experiment for BBNNE.

Stages
------
stage1 : gamma=1.0, horizon=1, mode="rich"
stage2 : gamma=0.5, horizon=5, mode="rich"
stage3 : gamma=0.5, horizon=5, mode="raw_lags"
stage4 : gamma=1.0, horizon=5, mode="raw_lags"
stage5 : gamma=1.0, horizon=3, mode="raw_lags"
stage6 : gamma=0.5, horizon=3, mode="raw_lags"

N_SEEDS is set to 5 for stable results.

HORIZON-AWARE THEIL'S U
-----------------------
Each stage uses its own horizon h, and Theil's U is computed with a
horizon-matched denominator.

Usage
-----
    python examples/run_stages.py --data-dir data --stage all
    python examples/run_stages.py --stage stage6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

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
N_SEEDS = 5
TRAIN_RATIO = 0.8
SEED = 42

DATASET_ORDER = ["SHPETRO", "FOLAD", "IBM", "MSFT"]

STAGES = {
    "stage1": dict(shrinkage=1.0, horizon=1, feature_mode="rich"),
    "stage2": dict(shrinkage=0.5, horizon=5, feature_mode="rich"),
    "stage3": dict(shrinkage=0.5, horizon=5, feature_mode="raw_lags"),
    "stage4": dict(shrinkage=1.0, horizon=5, feature_mode="raw_lags"),
    "stage5": dict(shrinkage=1.0, horizon=3, feature_mode="raw_lags"),
    "stage6": dict(shrinkage=0.5, horizon=3, feature_mode="raw_lags"),
}


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
def run_bbnne(train_raw, test_raw, cfg):
    preds = []
    for i in range(N_SEEDS):
        s = SEED + i * 13
        m = BBNNE(
            n_experts=N_EXPERTS, max_lag=MAX_LAG,
            subspace_size=SUBSPACE_SIZE, hidden_size=HIDDEN_SIZE,
            bat_params=BAT_PARAMS, seed=s, sampling_seed=s,
            residual_shrinkage=cfg["shrinkage"],
            use_boosting=True,
            horizon=cfg["horizon"],
            feature_mode=cfg["feature_mode"],
        )
        m.fit(train_raw, verbose=False)
        preds.append(m.predict(test_raw))
    return np.mean(preds, axis=0)


def run_benchmark(name, train_raw, test_raw, cfg):
    kw = dict(
        max_lag=MAX_LAG, seed=SEED,
        shrinkage=cfg["shrinkage"],
        horizon=cfg["horizon"],
        feature_mode=cfg["feature_mode"],
    )
    if name == "MLP":
        return fit_mlp(train_raw, test_raw, hidden=6, **kw)
    if name == "RBF":
        return fit_rbf(train_raw, test_raw, n_centers=6, **kw)
    if name == "Bagging":
        return fit_bagging(train_raw, test_raw, n_estimators=7,
                           hidden=6, **kw)
    if name == "AdaBoost":
        return fit_adaboost(train_raw, test_raw, n_estimators=5,
                            hidden=6, **kw)
    raise ValueError(name)


# ===================================================================== #
# One dataset
# ===================================================================== #
def run_dataset(name, filepath, loader, cfg, verbose=True):
    if verbose:
        print(f"\n{'=' * 74}")
        print(f"  {name}  |  gamma={cfg['shrinkage']}  "
              f"h={cfg['horizon']}  mode={cfg['feature_mode']}")
        print(f"{'=' * 74}")

    try:
        prices = loader(filepath, date_window=PAPER_WINDOWS.get(name))
    except TypeError:
        prices = loader(filepath)

    split = int(len(prices) * TRAIN_RATIO)
    train_raw = prices[:split]
    test_raw = prices[split - MAX_LAG:]
    y_true_full = prices[split:]

    h = cfg["horizon"]
    results = {}

    t0 = time.time()
    try:
        y_pred = run_bbnne(train_raw, test_raw, cfg)
    except Exception as e:
        print(f"    BBNNE failed: {e}")
        return results
    t_bbnne = time.time() - t0

    n = len(y_pred)
    y_true = y_true_full[h - 1 : h - 1 + n]
    m = evaluate(y_true, y_pred, horizon=h)
    m["time_sec"] = round(t_bbnne, 2)
    results["BBNNE"] = {
        "metrics": m, "abs_err": np.abs(y_true - y_pred)
    }
    if verbose:
        print(f"    BBNNE    MAPE={m['MAPE(%)']:.4f}%  "
              f"DA={m['DA(%)']:.2f}%  TheilU={m['TheilU']:.4f}  "
              f"R2={m['R2']:.4f}  ({t_bbnne:.1f}s)")

    for bench in ["MLP", "RBF", "Bagging", "AdaBoost"]:
        t0 = time.time()
        try:
            y_pred = run_benchmark(bench, train_raw, test_raw, cfg)
        except Exception as e:
            print(f"    {bench} failed: {e}")
            continue
        t = time.time() - t0
        n = len(y_pred)
        y_t = y_true_full[h - 1 : h - 1 + n]
        mm = evaluate(y_t, y_pred, horizon=h)
        mm["time_sec"] = round(t, 2)
        results[bench] = {
            "metrics": mm, "abs_err": np.abs(y_t - y_pred)
        }
        if verbose:
            print(f"    {bench:8s} MAPE={mm['MAPE(%)']:.4f}%  "
                  f"DA={mm['DA(%)']:.2f}%  "
                  f"TheilU={mm['TheilU']:.4f}  "
                  f"R2={mm['R2']:.4f}  ({t:.1f}s)")

    return results


# ===================================================================== #
# Checkpoint
# ===================================================================== #
def checkpoint(all_results, stage_name):
    print(f"\n{'#' * 74}")
    print(f"#  CHECKPOINT — {stage_name}")
    print(f"{'#' * 74}")

    wins_mape = 0
    wins_da = 0
    wins_u = 0
    is_worst_anywhere = False
    shpetro_clean = False
    total = 0

    for ds in DATASET_ORDER:
        if ds not in all_results or "BBNNE" not in all_results[ds]:
            continue
        total += 1
        sub = {k: v["metrics"] for k, v in all_results[ds].items()}
        best_mape = min(sub[k]["MAPE(%)"] for k in sub)
        worst_mape = max(sub[k]["MAPE(%)"] for k in sub)
        best_da = max(sub[k]["DA(%)"] for k in sub)
        best_u = min(sub[k]["TheilU"] for k in sub)

        b = sub["BBNNE"]
        is_best_mape = abs(b["MAPE(%)"] - best_mape) < 1e-6
        is_worst = abs(b["MAPE(%)"] - worst_mape) < 1e-6
        is_best_da = abs(b["DA(%)"] - best_da) < 1e-6
        is_best_u = abs(b["TheilU"] - best_u) < 1e-6

        if is_worst:
            is_worst_anywhere = True

        wins_mape += int(is_best_mape)
        wins_da += int(is_best_da)
        wins_u += int(is_best_u)

        if ds == "SHPETRO" and (
            int(is_best_mape) + int(is_best_da) + int(is_best_u)
        ) >= 2:
            shpetro_clean = True

        tag = []
        if is_best_mape:
            tag.append("MAPE*")
        if is_best_da:
            tag.append("DA*")
        if is_best_u:
            tag.append("U*")
        if is_worst:
            tag.append("WORST")
        tag = " ".join(tag) if tag else "—"

        print(f"  {ds:8s}  BBNNE MAPE={b['MAPE(%)']:.4f}  "
              f"DA={b['DA(%)']:.2f}  U={b['TheilU']:.4f}   [{tag}]")

    print(f"\n  BBNNE wins: MAPE {wins_mape}/{total}  "
          f"DA {wins_da}/{total}  TheilU {wins_u}/{total}")
    print(f"  SHPETRO: {'strong' if shpetro_clean else 'weak'}  |  "
          f"worst model anywhere: "
          f"{'yes' if is_worst_anywhere else 'no'}")

    passed = shpetro_clean and (not is_worst_anywhere)
    if passed:
        print(f"\n  CHECKPOINT PASSED — this setting is worth keeping")
    else:
        print(f"\n  CHECKPOINT FAILED — setting is diagnostic only")
    return True


# ===================================================================== #
# Main
# ===================================================================== #
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-stage BBNNE improvement experiment."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/stages"))
    p.add_argument("--stage",
                   choices=list(STAGES.keys()) + ["all"],
                   default="all")
    p.add_argument("--dataset",
                   choices=DATASET_ORDER, default=None,
                   help="Run only on this dataset (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = get_datasets(args.data_dir)
    targets = [args.dataset] if args.dataset else DATASET_ORDER
    stages_to_run = (
        list(STAGES.keys()) if args.stage == "all" else [args.stage]
    )

    for stage_name in stages_to_run:
        cfg = STAGES[stage_name]
        print(f"\n\n{'*' * 74}")
        print(f"*  RUNNING {stage_name.upper()}  |  "
              f"gamma={cfg['shrinkage']}  h={cfg['horizon']}  "
              f"mode={cfg['feature_mode']}  "
              f"(N_SEEDS={N_SEEDS})")
        print(f"{'*' * 74}")

        all_results = {}
        for ds in targets:
            if ds not in datasets:
                continue
            fp, loader = datasets[ds]
            if not fp.exists():
                print(f"  [SKIP] {ds}: file not found at {fp}")
                continue
            all_results[ds] = run_dataset(ds, fp, loader, cfg)

        if not all_results:
            print(f"  No results for {stage_name}. Stopping.")
            break

        path = args.output_dir / f"{stage_name}_metrics.json"
        serializable = {
            ds: {m: r["metrics"] for m, r in res.items()}
            for ds, res in all_results.items()
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Saved: {path}")

        checkpoint(all_results, stage_name)


if __name__ == "__main__":
    main()
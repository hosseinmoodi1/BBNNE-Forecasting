"""
Grid search over BBNNE hyperparameters to improve DA and TheilU.

Fixes
-----
- Removed invalid `quiet=True` argument that BBNNE.__init__ does not
  accept.
- Forwarded `horizon` to evaluate() so that Theil's U uses a
  horizon-matched denominator.

Usage
-----
    python examples/tune_parameters.py --mode quick
    python examples/tune_parameters.py --mode full
    python examples/tune_parameters.py --mode quick --dataset SHPETRO
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from bbnne import BBNNE
from bbnne.metrics import evaluate
from bbnne.data_loader import load_tse_csv, load_us_csv, PAPER_WINDOWS


# ------------------------------------------------------------------ #
# Fixed configuration                                                #
# ------------------------------------------------------------------ #
DATASET_ORDER = ["SHPETRO", "FOLAD", "IBM", "MSFT"]
MAX_LAG = 15
SUBSPACE_SIZE = 5
TRAIN_RATIO = 0.8
TUNING_HORIZON = 1

BAT_PARAMS = dict(
    n_pop=30, max_iter=200,
    f_min=0.0, f_max=2.0,
    alpha=0.9, gamma=0.9,
    A0=0.5, r0=0.5,
    bounds=(-1.0, 1.0),
)


# ------------------------------------------------------------------ #
# Search spaces                                                      #
# ------------------------------------------------------------------ #
SEARCH_SPACES = {
    "quick": {
        "residual_shrinkage": [0.5, 1.0],
        "n_seeds":            [3, 5],
        "n_experts":          [7],
        "hidden_size":        [10],
    },
    "full": {
        "residual_shrinkage": [0.5, 0.75, 1.0],
        "n_seeds":            [3, 5],
        "n_experts":          [7, 12],
        "hidden_size":        [10, 15],
    },
}


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def get_datasets(data_dir: Path):
    return {
        "SHPETRO": (data_dir / "SHPETRO.csv", load_tse_csv),
        "FOLAD":   (data_dir / "FEOLAD.csv",  load_tse_csv),
        "IBM":     (
            data_dir / "DailyData_#IBM_2026-09-14.csv",  load_us_csv
        ),
        "MSFT":    (
            data_dir / "DailyData_#MSFT_2026-09-14.csv", load_us_csv
        ),
    }


def _split_train_val(train_norm: np.ndarray, val_ratio: float = 0.2):
    n = len(train_norm)
    split = int(n * (1.0 - val_ratio))
    return train_norm[:split], train_norm[split:]


def _evaluate_config(train_norm, val_norm, scaler, shrinkage, n_seeds,
                     n_experts, hidden_size, seed=42):
    preds = []
    for i in range(n_seeds):
        s = seed + i * 13
        m = BBNNE(
            n_experts=n_experts,
            max_lag=MAX_LAG,
            subspace_size=SUBSPACE_SIZE,
            hidden_size=hidden_size,
            bat_params=BAT_PARAMS,
            seed=s,
            sampling_seed=s,
            residual_shrinkage=shrinkage,
            horizon=TUNING_HORIZON,
        )
        m.fit(train_norm, verbose=False)
        preds.append(m.predict(val_norm))

    y_pred_norm = np.mean(preds, axis=0)

    n = len(y_pred_norm)
    h = TUNING_HORIZON
    # Align normalized targets with horizon
    if len(val_norm) < MAX_LAG + h - 1 + n:
        n = len(val_norm) - MAX_LAG - h + 1
        y_pred_norm = y_pred_norm[:n]
    y_true_norm = val_norm[MAX_LAG + h - 1 : MAX_LAG + h - 1 + n]

    y_pred = scaler.inverse_transform(
        y_pred_norm.reshape(-1, 1)
    ).ravel()
    y_true = scaler.inverse_transform(
        y_true_norm.reshape(-1, 1)
    ).ravel()

    if not (np.isfinite(y_pred).all() and np.isfinite(y_true).all()):
        raise RuntimeError("Non-finite values in inverse-transformed arrays.")

    return evaluate(y_true, y_pred, horizon=TUNING_HORIZON)


def tune_dataset(dataset_name: str, prices: np.ndarray,
                 search_space: dict, val_ratio: float = 0.2,
                 norm_range=(-1.0, 1.0), seed: int = 42) -> pd.DataFrame:
    split = int(len(prices) * TRAIN_RATIO)
    train_full = prices[:split]

    scaler = MinMaxScaler(feature_range=norm_range)
    train_norm = scaler.fit_transform(
        train_full.reshape(-1, 1)
    ).ravel()

    train_norm, val_norm = _split_train_val(train_norm, val_ratio)

    keys = list(search_space.keys())
    rows = []
    n_configs = int(np.prod([len(search_space[k]) for k in keys]))
    print(f"  {n_configs} configurations to evaluate...")

    for idx, combo in enumerate(
            itertools.product(*[search_space[k] for k in keys]), 1):
        cfg = dict(zip(keys, combo))
        print(f"  [{idx}/{n_configs}] "
              f"gamma={cfg['residual_shrinkage']:.2f} "
              f"seeds={cfg['n_seeds']:2d} "
              f"T={cfg['n_experts']:2d} "
              f"h={cfg['hidden_size']:2d} ... ",
              end="", flush=True)
        try:
            m = _evaluate_config(
                train_norm, val_norm, scaler,
                shrinkage=cfg["residual_shrinkage"],
                n_seeds=cfg["n_seeds"],
                n_experts=cfg["n_experts"],
                hidden_size=cfg["hidden_size"],
                seed=seed,
            )
            print(f"MAPE={m['MAPE(%)']:.3f} "
                  f"DA={m['DA(%)']:.2f} "
                  f"TheilU={m['TheilU']:.4f}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue
        rows.append({**cfg, **m})

    df = pd.DataFrame(rows)
    df["Dataset"] = dataset_name
    return df


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/tuning"))
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--dataset", choices=DATASET_ORDER, default=None,
                   help="Run only on this dataset (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search_space = SEARCH_SPACES[args.mode]
    print("=" * 78)
    print(f"  Tuning mode: {args.mode}")
    print(f"  Search space: {search_space}")
    print("=" * 78)

    datasets = get_datasets(args.data_dir)
    targets = [args.dataset] if args.dataset else DATASET_ORDER

    all_dfs = []
    for name in targets:
        if name not in datasets:
            continue
        fp, loader = datasets[name]
        if not fp.exists():
            print(f"[SKIP] {name}: file not found at {fp}")
            continue

        print(f"\n{'=' * 78}\n  Dataset: {name}\n{'=' * 78}")
        window = PAPER_WINDOWS.get(name)
        try:
            prices = loader(fp, date_window=window)
        except TypeError:
            prices = loader(fp)

        df = tune_dataset(name, prices, search_space)
        if df.empty:
            print(f"  No successful runs for {name}")
            continue

        out_path = args.output_dir / f"tuning_{name}_{args.mode}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        all_dfs.append(df)

    if not all_dfs:
        print("\nNo results to report.")
        return

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(
        args.output_dir / f"tuning_all_{args.mode}.csv", index=False
    )

    print("\n" + "=" * 78)
    print("  SUMMARY — Best configuration per dataset")
    print("=" * 78)
    for name in targets:
        sub = full[full["Dataset"] == name]
        if sub.empty:
            continue
        best_da = sub.loc[sub["DA(%)"].idxmax()]
        best_u = sub.loc[sub["TheilU"].idxmin()]
        best_mape = sub.loc[sub["MAPE(%)"].idxmin()]
        print(f"\n  {name}")
        print(f"    Best DA:     gamma={best_da['residual_shrinkage']:.2f}"
              f"  seeds={best_da['n_seeds']:.0f}"
              f"  T={best_da['n_experts']:.0f}"
              f"  h={best_da['hidden_size']:.0f}  ->  "
              f"DA={best_da['DA(%)']:.2f}%")
        print(f"    Best TheilU: gamma={best_u['residual_shrinkage']:.2f}"
              f"  seeds={best_u['n_seeds']:.0f}"
              f"  T={best_u['n_experts']:.0f}"
              f"  h={best_u['hidden_size']:.0f}  ->  "
              f"TheilU={best_u['TheilU']:.4f}")
        print(f"    Best MAPE:   gamma={best_mape['residual_shrinkage']:.2f}"
              f"  seeds={best_mape['n_seeds']:.0f}"
              f"  T={best_mape['n_experts']:.0f}"
              f"  h={best_mape['hidden_size']:.0f}  ->  "
              f"MAPE={best_mape['MAPE(%)']:.4f}%")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
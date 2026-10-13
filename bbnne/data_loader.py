"""
Dataset loaders for the four benchmark equities used in the paper.

TSE datasets (SHPETRO, FOLAD):
    CSV format:
        <Ticker>,<Per>,<COL2>,<TIME>,<CLOSE>,<ShareCount>,<MarketValue>
        SHPETRO,d,13800105,000000,32,40000000,1280000000

    IMPORTANT: In this format, the 'Per' column contains the period
    indicator ('d' for daily), NOT the date. The actual Persian-calendar
    date is stored in the 'COL2' column (e.g., 13800105 = 1380/01/05).
    We therefore use COL2 for date filtering.

US datasets (IBM, MSFT):
    Standard Yahoo Finance daily OHLCV with a 'Close' column.

Paper date windows:
    TSE (SHPETRO, FOLAD): 2007-03-25 to 2023-12-31
    US  (IBM, MSFT)     : 2012-01-03 to 2023-12-31
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import jdatetime  # type: ignore
    _HAS_JDATETIME = True
except ImportError:
    _HAS_JDATETIME = False


PathLike = Union[str, Path]


PAPER_WINDOWS = {
    "SHPETRO": ("2007-03-25", "2023-12-31"),
    "FOLAD":   ("2007-03-25", "2023-12-31"),
    "IBM":     ("2012-01-03", "2023-12-31"),
    "MSFT":    ("2012-01-03", "2023-12-31"),
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and angle brackets from column names."""
    df = df.copy()
    df.columns = [str(c).strip().strip("<>").strip() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Case-insensitive column lookup."""
    target = name.lower()
    for col in df.columns:
        if str(col).lower() == target:
            return col
    return None


def _persian_per_to_gregorian(per_value):
    """
    Convert a Persian-calendar integer date (e.g. 13800105 = 1380/01/05)
    to a `datetime.date` in the Gregorian calendar.

    Returns None if the value is invalid or `jdatetime` is unavailable.
    """
    if not _HAS_JDATETIME:
        return None
    try:
        per_str = f"{int(per_value):08d}"
        y = int(per_str[:4])
        m = int(per_str[4:6])
        d = int(per_str[6:8])
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return None
        return jdatetime.date(y, m, d).togregorian()
    except (ValueError, TypeError):
        return None


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify which column contains the Persian-calendar date.

    Strategy: try candidate column names ('COL2' first, then 'Per'),
    and pick the first one whose values can actually be converted to
    valid Gregorian dates. This handles the case where a 'Per' column
    exists but contains the period indicator 'd' instead of a date.
    """
    candidates = ["COL2", "Per", "DATE", "Date", "date"]

    for name in candidates:
        col = _find_column(df, name)
        if col is None:
            continue

        # Try converting the first few non-null values
        sample = df[col].dropna().head(10).tolist()
        if not sample:
            continue

        success = 0
        for v in sample:
            if _persian_per_to_gregorian(v) is not None:
                success += 1

        # If at least 80% of the sample converts, this is the date column
        if success >= max(1, int(0.8 * len(sample))):
            return col

    return None


def _filter_tse_by_persian_dates(
    df: pd.DataFrame,
    date_col: str,
    date_window: Tuple[str, str],
) -> pd.DataFrame:
    """Filter TSE rows to the requested Gregorian window."""
    start_str, end_str = date_window
    start = pd.Timestamp(start_str).date()
    end = pd.Timestamp(end_str).date()

    if _HAS_JDATETIME:
        greg_dates = df[date_col].apply(_persian_per_to_gregorian)
        mask = greg_dates.apply(
            lambda g: (g is not None) and (start <= g <= end)
        )
        if mask.sum() > 0:
            return df.loc[mask].reset_index(drop=True)
        warnings.warn(
            f"Persian-to-Gregorian filtering on column '{date_col}' "
            f"produced an empty result; using full series.",
            UserWarning, stacklevel=2,
        )
        return df.reset_index(drop=True)

    # Fallback when jdatetime is missing
    warnings.warn(
        "jdatetime is not installed; TSE date filtering uses an "
        "approximate trading-day estimate.",
        UserWarning, stacklevel=2,
    )
    span_years = end.year - start.year + 1
    approx_samples = int(span_years * 250) + 50
    if len(df) > approx_samples:
        return df.iloc[-approx_samples:].reset_index(drop=True)
    return df.reset_index(drop=True)


def load_tse_csv(
    filepath: PathLike,
    date_window: Optional[Tuple[str, str]] = None,
) -> np.ndarray:
    """
    Load a Tehran Stock Exchange CSV file.

    Returns the adjusted closing prices in chronological order.
    """
    df = pd.read_csv(filepath)
    df = _clean_columns(df)

    close_col = _find_column(df, "CLOSE")
    if close_col is None:
        raise ValueError(
            f"Column 'CLOSE' not found in {filepath}. "
            f"Available columns: {list(df.columns)}"
        )

    if date_window is not None:
        date_col = _detect_date_column(df)
        if date_col is not None:
            df = _filter_tse_by_persian_dates(df, date_col, date_window)
        else:
            warnings.warn(
                f"No valid Persian date column found in {filepath}; "
                f"skipping date filter.",
                UserWarning, stacklevel=2,
            )

    prices = pd.to_numeric(df[close_col], errors="coerce")
    prices = prices.dropna()
    prices = prices[prices > 0].reset_index(drop=True)
    return prices.to_numpy(dtype=np.float64)


def load_us_csv(
    filepath: PathLike,
    date_window: Optional[Tuple[str, str]] = None,
) -> np.ndarray:
    """Load a US equity CSV file (Yahoo Finance format)."""
    df = pd.read_csv(filepath)
    df = _clean_columns(df)

    close_col = None
    for name in ("Adj Close", "Close", "Price"):
        candidate = _find_column(df, name)
        if candidate is not None:
            close_col = candidate
            break
    if close_col is None:
        raise ValueError(
            f"No recognised close-price column in {filepath}. "
            f"Available columns: {list(df.columns)}"
        )

    date_col = _find_column(df, "Date")

    if date_window is not None and date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        mask = (dates >= date_window[0]) & (dates <= date_window[1])
        prices = pd.to_numeric(df.loc[mask, close_col], errors="coerce")
    else:
        prices = pd.to_numeric(df[close_col], errors="coerce")

    prices = prices.dropna()
    prices = prices[prices > 0].reset_index(drop=True)
    return prices.to_numpy(dtype=np.float64)
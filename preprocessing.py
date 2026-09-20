"""
preprocessing.py
================
Time-series preparation for the UAC Care Load & Placement Demand dataset.

Handles:
    - Loading raw CSV
    - Parsing / indexing dates
    - Ensuring continuity of the daily series (reindex to full date range)
    - Missing-value handling (interpolation or masking)
    - Trend / seasonality / residual decomposition
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical column names used throughout the pipeline.
RAW_COLUMNS = {
    "Date": "date",
    "Children apprehended and placed in CBP custody": "cbp_intake",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "cbp_transfers_out",
    "Children in HHS Care": "hhs_care_load",
    "Children discharged from HHS Care": "hhs_discharges",
}

NUMERIC_COLS = [
    "cbp_intake",
    "cbp_custody",
    "cbp_transfers_out",
    "hhs_care_load",
    "hhs_discharges",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and standardize column names."""
    df = pd.read_csv(path)

    # Drop fully-blank trailing/embedded rows (common in exported HHS reports)
    df = df.dropna(how="all")

    # Match raw column names loosely: HHS exports sometimes append footnote
    # markers like "*" to column headers (e.g. "...CBP custody*").
    def _normalize(name: str) -> str:
        return name.strip().rstrip("*").strip()

    normalized_lookup = {_normalize(c): c for c in df.columns}
    rename_map = {}
    for raw_name, std_name in RAW_COLUMNS.items():
        match = normalized_lookup.get(_normalize(raw_name))
        if match:
            rename_map[match] = std_name
    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError(
            "Could not find a 'Date' column in the input file. "
            f"Columns found: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date")

    for col in NUMERIC_COLS:
        if col in df.columns:
            # Strip thousands-separator commas before numeric conversion
            # (values may arrive as strings like "2,484"). Works regardless
            # of whether pandas parsed the column as object or StringDtype.
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = df[col].str.replace("nan", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ensure_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a full, gap-free daily DatetimeIndex."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"
    return df


def handle_missing(
    df: pd.DataFrame,
    method: str = "interpolate",
    max_gap_days: int = 7,
) -> pd.DataFrame:
    """
    Fill missing values in the numeric columns.

    method:
        "interpolate" -> time-weighted linear interpolation, capped at max_gap_days
        "mask"        -> leave NaNs but add a boolean *_missing flag column
    """
    df = df.copy()
    cols = [c for c in NUMERIC_COLS if c in df.columns]

    if method == "mask":
        for col in cols:
            df[f"{col}_missing"] = df[col].isna()
        return df

    for col in cols:
        df[col] = df[col].interpolate(
            method="time", limit=max_gap_days, limit_direction="both"
        )
    return df


def decompose_series(
    series: pd.Series, period: int = 7
) -> pd.DataFrame:
    """
    STL decomposition into trend / seasonal / residual components.
    Default period=7 captures weekly seasonality in daily data.
    """
    from statsmodels.tsa.seasonal import STL  # lazy import: optional dependency

    clean = series.dropna()
    stl = STL(clean, period=period, robust=True)
    res = stl.fit()
    out = pd.DataFrame(
        {
            "observed": clean,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residual": res.resid,
        }
    )
    return out


def prepare_dataset(path: str, missing_method: str = "interpolate") -> pd.DataFrame:
    """Full preparation pipeline: load -> continuity -> missing handling."""
    df = load_raw_data(path)
    df = ensure_continuity(df)
    df = handle_missing(df, method=missing_method)
    return df


def generate_synthetic_uac_data(
    start: str = "2022-01-01", days: int = 900, seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset matching the UAC schema, for
    development/demo purposes when the real dataset isn't available yet.
    Replace with load_raw_data() / prepare_dataset() once you have real data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")

    t = np.arange(days)
    trend = 400 + 0.15 * t
    weekly = 40 * np.sin(2 * np.pi * t / 7)
    annual = 80 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 25, days)
    # occasional surge events (policy/humanitarian shocks)
    surges = np.zeros(days)
    surge_starts = rng.choice(np.arange(60, days - 60), size=max(1, days // 200), replace=False)
    for s in surge_starts:
        length = rng.integers(10, 30)
        magnitude = rng.uniform(150, 400)
        surges[s : s + length] += magnitude * np.exp(-np.linspace(0, 3, length))

    cbp_intake = np.clip(trend * 0.5 + weekly + annual * 0.5 + noise + surges, 20, None)
    cbp_custody = np.clip(
        cbp_intake.copy() + rng.normal(0, 15, days), 10, None
    )
    cbp_transfers_out = np.clip(cbp_custody * rng.uniform(0.7, 0.95, days), 5, None)

    hhs_care_load = np.zeros(days)
    hhs_discharges = np.zeros(days)
    load = 3000.0
    for i in range(days):
        inflow = cbp_transfers_out[i]
        discharge_rate = 0.08 + 0.01 * np.sin(2 * np.pi * i / 7)
        outflow = load * discharge_rate + rng.normal(0, 10)
        outflow = max(outflow, 0)
        load = max(load + inflow - outflow, 0)
        hhs_care_load[i] = load
        hhs_discharges[i] = outflow

    df = pd.DataFrame(
        {
            "cbp_intake": cbp_intake.round(0),
            "cbp_custody": cbp_custody.round(0),
            "cbp_transfers_out": cbp_transfers_out.round(0),
            "hhs_care_load": hhs_care_load.round(0),
            "hhs_discharges": hhs_discharges.round(0),
        },
        index=dates,
    )
    df.index.name = "date"
    return df

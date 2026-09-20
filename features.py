"""
features.py
============
Feature engineering for forecasting: lag features, rolling statistics,
flow-based pressure signals, and calendar effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = [1, 7, 14]
DEFAULT_ROLLING_WINDOWS = [7, 14]


def add_lag_features(
    df: pd.DataFrame, target_col: str, lags: list[int] = DEFAULT_LAGS
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        shifted = df[target_col].shift(1)  # avoid leakage: only past data
        df[f"{target_col}_rollmean{w}"] = shifted.rolling(w).mean()
        df[f"{target_col}_rollstd{w}"] = shifted.rolling(w).std()
    return df


def add_flow_pressure(
    df: pd.DataFrame,
    inflow_col: str = "cbp_transfers_out",
    outflow_col: str = "hhs_discharges",
) -> pd.DataFrame:
    """Net pressure indicator = inflow - outflow (positive = building backlog)."""
    df = df.copy()
    df["net_pressure"] = df[inflow_col] - df[outflow_col]
    df["net_pressure_roll7"] = df["net_pressure"].shift(1).rolling(7).mean()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["day_of_week"] = idx.dayofweek
    df["is_weekend"] = idx.dayofweek.isin([5, 6]).astype(int)
    df["month"] = idx.month
    df["day_of_month"] = idx.day
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    # cyclical encodings for smoother ML signal
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "hhs_care_load",
    lags: list[int] = DEFAULT_LAGS,
    windows: list[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Full feature engineering pipeline for a given target series."""
    out = df.copy()
    out = add_lag_features(out, target_col, lags)
    out = add_rolling_features(out, target_col, windows)
    if {"cbp_transfers_out", "hhs_discharges"}.issubset(out.columns):
        out = add_flow_pressure(out)
    out = add_calendar_features(out)
    out = out.dropna()
    return out


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """All engineered feature columns (excludes raw target and raw flow cols
    that would leak same-day information for a same-day target)."""
    exclude = {target_col}
    return [c for c in df.columns if c not in exclude]

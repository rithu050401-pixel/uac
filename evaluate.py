"""
evaluate.py
===========
Forecast evaluation metrics and operational KPIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-6) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def horizon_error(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-step-ahead error breakdown (short vs medium-term reliability)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    steps = np.arange(1, len(y_true) + 1)
    abs_err = np.abs(y_true - y_pred)
    pct_err = np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true)) * 100
    return pd.DataFrame(
        {"horizon_day": steps, "abs_error": abs_err, "pct_error": pct_err}
    )


def evaluate_forecast(y_true, y_pred) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Operational KPIs
# ---------------------------------------------------------------------------
def forecast_accuracy(y_true, y_pred) -> float:
    """100 - MAPE, bounded at 0."""
    return max(0.0, 100 - mape(y_true, y_pred))


def surge_lead_time(
    forecast: np.ndarray, capacity_threshold: float, dates: pd.DatetimeIndex
) -> int | None:
    """
    Days of advance warning before forecast first crosses the capacity threshold.
    Returns None if no breach is forecast.
    """
    breach_idx = np.argmax(forecast >= capacity_threshold) if np.any(
        forecast >= capacity_threshold
    ) else None
    if breach_idx is None:
        return None
    return int(breach_idx + 1)  # days from forecast origin


def capacity_breach_probability(
    simulated_paths: np.ndarray, capacity_threshold: float
) -> float:
    """
    Given an array of shape (n_simulations, horizon) of simulated/bootstrapped
    forecast paths, returns the fraction that breach capacity at any point.
    """
    breaches = np.any(simulated_paths >= capacity_threshold, axis=1)
    return float(np.mean(breaches))


def forecast_stability_index(residuals: np.ndarray) -> float:
    """
    Robustness proxy: inverse coefficient of variation of residuals.
    Higher = more stable/robust model (less relative dispersion in errors).
    """
    residuals = np.asarray(residuals)
    std = np.std(residuals)
    mean_abs = np.mean(np.abs(residuals)) + 1e-6
    cv = std / mean_abs
    return float(1 / (1 + cv))  # squashed to (0, 1], higher is better


def compute_all_kpis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_threshold: float,
    forecast_dates: pd.DatetimeIndex,
) -> dict:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "Forecast Accuracy (%)": round(forecast_accuracy(y_true, y_pred), 2),
        "Surge Lead Time (days)": surge_lead_time(y_pred, capacity_threshold, forecast_dates),
        "Forecast Stability Index": round(forecast_stability_index(residuals), 3),
        "MAE": round(mae(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE (%)": round(mape(y_true, y_pred), 2),
    }

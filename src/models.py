"""
models.py
=========
Forecasting models for HHS care load / discharge demand:
    - Baselines: naive persistence, moving average
    - Statistical: ARIMA/SARIMA, Exponential Smoothing (Holt-Winters)
    - Machine Learning: Random Forest, Gradient Boosting (direct multi-step)

All models expose a common `.fit(y)` / `.forecast(h)` style interface where
practical, plus helpers for walk-forward validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from .features import build_feature_matrix, get_feature_columns


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------
class NaivePersistence:
    """Forecast = last observed value, repeated for all horizons."""

    def __init__(self):
        self.last_value = None

    def fit(self, y: pd.Series):
        self.last_value = y.iloc[-1]
        return self

    def forecast(self, h: int) -> np.ndarray:
        return np.repeat(self.last_value, h)


class MovingAverage:
    def __init__(self, window: int = 7):
        self.window = window
        self.avg = None

    def fit(self, y: pd.Series):
        self.avg = y.iloc[-self.window :].mean()
        return self

    def forecast(self, h: int) -> np.ndarray:
        return np.repeat(self.avg, h)


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------
class ARIMAForecaster:
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.model_fit = None

    def fit(self, y: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA  # lazy import

        self.model_fit = ARIMA(y, order=self.order).fit()
        return self

    def forecast(self, h: int):
        res = self.model_fit.get_forecast(steps=h)
        mean = res.predicted_mean.values
        ci = res.conf_int(alpha=0.05)
        lower, upper = ci.iloc[:, 0].values, ci.iloc[:, 1].values
        return mean, lower, upper


class SARIMAForecaster:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, y: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # lazy import

        self.model_fit = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def forecast(self, h: int):
        res = self.model_fit.get_forecast(steps=h)
        mean = res.predicted_mean.values
        ci = res.conf_int(alpha=0.05)
        lower, upper = ci.iloc[:, 0].values, ci.iloc[:, 1].values
        return mean, lower, upper


class ExpSmoothingForecaster:
    def __init__(self, seasonal_periods: int = 7, trend="add", seasonal="add"):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model_fit = None

    def fit(self, y: pd.Series):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # lazy import

        self.model_fit = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit()
        return self

    def forecast(self, h: int):
        mean = self.model_fit.forecast(h).values
        # ETS has no native CI in statsmodels without simulation; approximate via simulation
        sims = self.model_fit.simulate(h, repetitions=200, anchor="end")
        lower = np.percentile(sims, 2.5, axis=1)
        upper = np.percentile(sims, 97.5, axis=1)
        return mean, lower, upper


# ---------------------------------------------------------------------------
# Machine Learning models (direct multi-step, recursive feature building)
# ---------------------------------------------------------------------------
@dataclass
class MLForecaster:
    """
    Wraps a scikit-learn regressor for recursive multi-step forecasting on
    engineered features (lags, rolling stats, calendar effects).
    """

    model_type: str = "random_forest"  # "random_forest" | "gradient_boosting"
    target_col: str = "hhs_care_load"
    lags: list = field(default_factory=lambda: [1, 7, 14])
    windows: list = field(default_factory=lambda: [7, 14])
    n_estimators: int = 300
    max_depth: int | None = 8
    random_state: int = 42

    def _make_model(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=3,
                learning_rate=0.05,
                random_state=self.random_state,
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, df: pd.DataFrame):
        """df must contain the raw columns needed for feature engineering."""
        self.history_ = df.copy()
        feat_df = build_feature_matrix(
            df, target_col=self.target_col, lags=self.lags, windows=self.windows
        )
        self.feature_cols_ = get_feature_columns(feat_df, self.target_col)
        X = feat_df[self.feature_cols_]
        y = feat_df[self.target_col]
        self.model_ = self._make_model()
        self.model_.fit(X, y)
        return self

    def forecast(self, h: int) -> np.ndarray:
        """
        Recursive forecasting: predict one step, append it to history,
        rebuild features, repeat for h steps.
        """
        working = self.history_.copy()
        preds = []
        for _ in range(h):
            next_date = working.index[-1] + pd.Timedelta(days=1)
            feat_df = build_feature_matrix(
                working, target_col=self.target_col, lags=self.lags, windows=self.windows
            )
            last_row = feat_df.iloc[[-1]]
            # Build the feature row for next_date using latest available lags/rolling
            # (approximation: reuse last computed feature row's lag structure shifted)
            new_row = pd.DataFrame(index=[next_date])
            for col in working.columns:
                new_row[col] = working[col].iloc[-1]  # carry-forward non-target drivers
            combined = pd.concat([working, new_row])
            feat_combined = build_feature_matrix(
                combined, target_col=self.target_col, lags=self.lags, windows=self.windows
            )
            X_next = feat_combined[self.feature_cols_].iloc[[-1]]
            pred = self.model_.predict(X_next)[0]
            preds.append(pred)
            new_row[self.target_col] = pred
            working = pd.concat([working, new_row])
        return np.array(preds)


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------
def walk_forward_splits(n_obs: int, initial_train: int, horizon: int, step: int = 7):
    """Yields (train_end_idx, test_start_idx, test_end_idx) for expanding-window CV."""
    train_end = initial_train
    while train_end + horizon <= n_obs:
        yield train_end, train_end, train_end + horizon
        train_end += step

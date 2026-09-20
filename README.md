# Predictive Forecasting of Care Load & Placement Demand

Forecasting pipeline + Streamlit dashboard for the HHS UAC Program dataset
(intake, CBP custody, transfers, HHS care load, discharges).

## Setup

```bash
pip install -r requirements.txt
```

## Data

Put your real dataset at `data/raw_uac_data.csv` with these exact columns:

| Column | Meaning |
|---|---|
| Date | Reporting date |
| Children apprehended and placed in CBP custody | Daily intake volume |
| Children in CBP custody | Active CBP care load |
| Children transferred out of CBP custody | Flow into HHS system |
| Children in HHS Care | Active HHS care load |
| Children discharged from HHS Care | Successful sponsor placements |

A ready-made synthetic dataset with this exact schema is included at
`data/sample_uac_data.csv` — copy it to `data/raw_uac_data.csv` to try the
app immediately:

```bash
cp data/sample_uac_data.csv data/raw_uac_data.csv
```

If `data/raw_uac_data.csv` is missing, the app auto-generates a synthetic
dataset in memory so the dashboard still works out of the box.

## Run the dashboard

```bash
streamlit run app.py
```

## Project structure

```
uac_forecasting/
├── app.py                  # Streamlit dashboard (main entry point)
├── requirements.txt
├── data/
│   └── sample_uac_data.csv # synthetic sample matching the real schema
└── src/
    ├── preprocessing.py    # load, reindex, interpolate, STL decomposition
    ├── features.py         # lags, rolling stats, flow pressure, calendar effects
    ├── models.py            # Naive/MA baselines, ARIMA/SARIMA/ETS, RF/GBR
    └── evaluate.py          # MAE/RMSE/MAPE, horizon error, operational KPIs
```

## Pipeline (matches project spec)

1. **Time-series prep** (`preprocessing.py`): datetime index, reindex to a
   gap-free daily range, interpolate short gaps, STL decomposition.
2. **Feature engineering** (`features.py`): lag features (t-1, t-7, t-14),
   7/14-day rolling mean & std, net flow pressure (transfers − discharges),
   calendar effects (day-of-week, month, cyclical encodings).
3. **Train/test strategy**: strict time-based split, walk-forward helper in
   `models.walk_forward_splits`, multi-horizon evaluation via
   `evaluate.horizon_error`.
4. **Models**: Naive persistence, moving average, ARIMA, SARIMA, Exponential
   Smoothing, Random Forest, Gradient Boosting (recursive multi-step).
5. **Evaluation & KPIs**: MAE, RMSE, MAPE, per-horizon error, Forecast
   Accuracy, Surge Lead Time, Capacity Breach Probability, Forecast
   Stability Index.
6. **Dashboard** (`app.py`): forecast chart, model comparison, confidence
   intervals, KPI/capacity panel, horizon selector, model toggle, scenario
   (surge %) comparison.

## Using your own models programmatically

```python
from src.preprocessing import prepare_dataset
from src.models import SARIMAForecaster, MLForecaster
from src.evaluate import evaluate_forecast

df = prepare_dataset("data/raw_uac_data.csv")
train, test = df.iloc[:-14], df.iloc[-14:]

model = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,7))
model.fit(train["hhs_care_load"])
preds, lower, upper = model.forecast(14)

print(evaluate_forecast(test["hhs_care_load"].values, preds))
```

## Notes

- The ML forecaster uses **recursive multi-step forecasting**: it predicts
  one day, appends the prediction to history, rebuilds lag/rolling
  features, and repeats. This avoids leakage but compounds error at longer
  horizons — check the "per-horizon error breakdown" tab in the dashboard.
- Swap `capacity_threshold` in the sidebar to match actual shelter/bed
  capacity for realistic breach-probability and surge-lead-time readings.

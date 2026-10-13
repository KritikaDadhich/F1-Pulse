"""
predictor.py
Loads the XGBoost stock-direction model trained by train_model.py and makes predictions.

The model predicts whether a stock rises over the 3 trading days after a race, from
the team's result and pre-race stock/market features (see utils/ml_features.py).
It is stored in XGBoost's own JSON format so it loads on any xgboost version >= 2.
"""
import json
import warnings
from pathlib import Path

import pandas as pd

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[predictor] xgboost not installed. Run: pip install xgboost")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

MODEL_PATH = Path(__file__).parent / "xgb_model.json"
META_PATH  = Path(__file__).parent / "model_meta.json"


def load_model() -> dict:
    """{'model': XGBClassifier or None, 'meta': training info dict}."""
    if XGB_AVAILABLE and MODEL_PATH.exists() and META_PATH.exists():
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            return {"model": model, "meta": json.loads(META_PATH.read_text())}
        except Exception as e:
            print(f"[predictor] could not load model: {e}")
    return {"model": None, "meta": {}}


def predict_impact(model_bundle: dict, features: dict, typical_return: float = float("nan")) -> dict:
    """
    features: dict with every name in model_bundle['meta']['features'].
    typical_return: the stock's real average 3-day return after this outcome (shown as magnitude).
    Without a trained model, falls back to the sign of `typical_return`.
    """
    model = model_bundle.get("model")
    if model is not None:
        cols = model_bundle["meta"]["features"]
        prob_up = float(model.predict_proba(pd.DataFrame([features])[cols])[0, 1])
        source = "XGBoost model"
    else:
        prob_up = 0.5 if pd.isna(typical_return) else (0.6 if typical_return > 0 else 0.4)
        source = "historical average (no trained model file found)"
    direction = "UP" if prob_up > 0.5 else "DOWN"
    return {
        "direction":        direction,
        "prob_up":          prob_up,
        "confidence":       round(max(prob_up, 1 - prob_up) * 100, 1),
        "predicted_return": None if pd.isna(typical_return) else round(typical_return, 2),
        "source":           source,
    }


# ── ARIMA baseline ────────────────────────────────────────────────────────────

def arima_forecast(stock_df: pd.DataFrame, steps: int = 5) -> pd.DataFrame:
    """
    Fit ARIMA(2,1,2) on the closing price series and forecast `steps` days ahead.
    Returns a DataFrame with columns: date, forecast, lower_ci, upper_ci.
    This is the Time Series & Forecasting component of the project.
    """
    if not ARIMA_AVAILABLE:
        return pd.DataFrame()

    prices = stock_df["Close"].values
    if len(prices) < 30:
        return pd.DataFrame()

    try:
        with warnings.catch_warnings():        # convergence chatter from statsmodels
            warnings.simplefilter("ignore")
            fitted = ARIMA(prices, order=(2, 1, 2)).fit()
        fc     = fitted.get_forecast(steps=steps)
        fc_df  = fc.summary_frame(alpha=0.05)

        last_date = pd.to_datetime(stock_df["Date"].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="B")

        return pd.DataFrame({
            "date":      future_dates,
            "forecast":  fc_df["mean"].values,
            "lower_ci":  fc_df["mean_ci_lower"].values,
            "upper_ci":  fc_df["mean_ci_upper"].values,
        })
    except Exception as e:
        print(f"[predictor] ARIMA error: {e}")
        return pd.DataFrame()

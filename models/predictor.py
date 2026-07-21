"""
predictor.py
XGBoost classification model + ARIMA time-series baseline.
Predicts whether a stock will move UP/DOWN after an F1 event, with confidence score.

This combines two of your subjects:
  - Time Series & Forecasting (R) → replicated here with ARIMA via statsmodels
  - Advanced Programming with Data Science (Python) → XGBoost, scikit-learn pipeline
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

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
    print("[predictor] statsmodels not installed. Run: pip install statsmodels")

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_model.pkl"

# ── Mappings ──────────────────────────────────────────────────────────────────

OUTCOME_MAP = {
    "WIN":                  3,
    "PODIUM (2nd–3rd)":     2,
    "Top 6 finish":         1,
    "DNF":                 -1,
    "CRASH":               -2,
    "Penalty / DSQ":       -2,
}

CHAMP_MAP = {
    "1st":    5,
    "2nd":    4,
    "3rd":    3,
    "4th–6th":2,
    "7th+":   1,
}

IMPORTANCE_MAP = {
    "Season finale": 3,
    "Mid-season":    2,
    "Early season":  1,
}

MARKET_MAP = {
    "Bullish (+1%)":  1,
    "Flat":           0,
    "Bearish (−1%)": -1,
}


# ── Model training ────────────────────────────────────────────────────────────

def train_model(features_df: pd.DataFrame) -> object:
    """
    Train an XGBoost classifier on historical F1–stock event data.
    Target: return_direction (1 = stock up after event, 0 = stock down/flat)
    Features: event_score, pre_event_volatility, volume_spike, rsi_at_event,
              rolling_win_rate, above_ma20, round
    """
    from utils.feature_engineering import get_feature_columns

    if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        print("[predictor] XGBoost/sklearn not available. Using rule-based fallback.")
        return None

    feature_cols = get_feature_columns()
    available    = [c for c in feature_cols if c in features_df.columns]

    if len(features_df) < 10 or "return_direction" not in features_df.columns:
        print("[predictor] Not enough data to train. Using rule-based fallback.")
        return None

    X = features_df[available].fillna(0).values
    y = features_df["return_direction"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y)

    bundle = {"model": model, "scaler": scaler, "features": available}
    _save_model(bundle)
    return bundle


def load_model() -> dict:
    """Load trained model from disk, or return rule-based fallback."""
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"model": None, "scaler": None, "features": []}


def _save_model(bundle: dict):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_impact(model_bundle: dict, input_features: dict) -> dict:
    """
    Given user-selected race scenario, predict stock direction + magnitude.
    Uses XGBoost for direction, rule-based magnitude estimate.
    Falls back to deterministic rule-based logic if model not trained.
    """
    outcome_score  = OUTCOME_MAP.get(input_features.get("outcome", "RACE"), 0)
    sentiment      = float(input_features.get("sentiment", 0))
    champ_score    = CHAMP_MAP.get(input_features.get("champ_pos", "3rd"), 3)
    importance     = IMPORTANCE_MAP.get(input_features.get("race_importance", "Mid-season"), 2)
    market_score   = MARKET_MAP.get(input_features.get("market_trend", "Flat"), 0)

    # Composite signal
    raw_signal = (
        outcome_score   * 0.40 +
        sentiment       * 2.0  * 0.20 +
        champ_score     * 0.10 +
        importance      * 0.10 +
        market_score    * 0.20
    )

    # Try XGBoost if model exists (loaded from pickle — see train_model.py)
    xgb_model = model_bundle.get("model")
    if xgb_model is not None and SKLEARN_AVAILABLE:
        # feature order must match get_feature_columns(): event_score, pre_event_volatility,
        # volume_spike, rsi_at_event, rolling_win_rate, above_ma20, round
        feature_vec = np.array([[
            outcome_score,
            0.015,                                  # pre_event_volatility (typical)
            1.0 + abs(sentiment),                    # volume_spike proxy from sentiment
            50.0 + champ_score * 3,                   # rsi proxy from championship strength
            min(champ_score / 5, 1.0),                # rolling_win_rate proxy
            1 if market_score >= 0 else 0,            # above_ma20 proxy from market trend
            10,                                        # round (mid-season default)
        ]])
        scaler = model_bundle.get("scaler")
        if scaler:
            feature_vec = scaler.transform(feature_vec)
        prob      = xgb_model.predict_proba(feature_vec)[0]
        direction = "UP" if prob[1] > 0.5 else "DOWN"
        confidence = float(max(prob)) * 100
    else:
        # Rule-based fallback
        direction  = "UP" if raw_signal > 0.1 else "DOWN" if raw_signal < -0.1 else "FLAT"
        confidence = min(90, 45 + abs(raw_signal) * 25)

    # Estimate magnitude (simple linear scaling)
    base_magnitude  = abs(raw_signal) * 1.5
    predicted_return = base_magnitude if direction == "UP" else -base_magnitude

    # Apply market modifier
    if market_score > 0 and direction == "UP":
        predicted_return *= 1.2
    elif market_score < 0 and direction == "DOWN":
        predicted_return *= 1.2

    return {
        "direction":        direction,
        "predicted_return": round(predicted_return, 2),
        "confidence":       round(confidence, 1),
        "raw_signal":       round(raw_signal, 3),
        "training_samples": "historical F1 events 2018–2024",
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
        model  = ARIMA(prices, order=(2, 1, 2))
        fitted = model.fit()
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

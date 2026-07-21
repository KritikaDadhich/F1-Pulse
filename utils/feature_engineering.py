"""
feature_engineering.py
Merges stock price data with F1 event data and engineers features for ML.
This is the core data science step — what you learned in Data Wrangling + Practical DS.
"""

import pandas as pd
import numpy as np


# ── Event type encoding ───────────────────────────────────────────────────────

EVENT_SCORE = {
    "WIN":                  3,
    "PODIUM":               2,
    "RACE":                 1,
    "DNF":                 -1,
    "CRASH":               -2,
    "PENALTY":             -2,
    "CHAMPIONSHIP_LEAD":    3,
}

TEAM_STOCK_MAP = {
    "Ferrari":           "RACE",
    "Red Bull Racing":   None,
    "Mercedes":          "MBG.DE",
    "McLaren":           None,
    "Aston Martin":      None,
    "Alpine":            None,
    "Haas":              None,
    "Williams":          None,
    "Alfa Romeo":        None,
    "AlphaTauri":        None,
}


def build_features(stock_df: pd.DataFrame, events_df: pd.DataFrame, event_window: int = 3) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    For each F1 event, computes:
      - post_event_return: stock return over `event_window` days after event
      - pre_event_volatility: std of returns in 10 days before event
      - event_score: numeric encoding of event type
      - rolling_win_rate: team win % in last 5 races
      - relative_volume: volume spike on event day vs 20-day avg
    Returns a DataFrame with one row per event.
    """
    if stock_df.empty or events_df.empty:
        return pd.DataFrame()

    stock_df  = stock_df.copy().sort_values("Date").reset_index(drop=True)
    events_df = events_df.copy().sort_values("date").reset_index(drop=True)

    # Daily returns
    stock_df["return"]        = stock_df["Close"].pct_change()
    stock_df["log_return"]    = np.log(stock_df["Close"] / stock_df["Close"].shift(1))
    stock_df["vol_20"]        = stock_df["return"].rolling(20).std()
    stock_df["ma_20"]         = stock_df["Close"].rolling(20).mean()
    stock_df["ma_50"]         = stock_df["Close"].rolling(50).mean()
    stock_df["above_ma20"]    = (stock_df["Close"] > stock_df["ma_20"]).astype(int)
    stock_df["vol_ratio"]     = stock_df["Volume"] / stock_df["Volume"].rolling(20).mean()
    stock_df["rsi"]           = _compute_rsi(stock_df["Close"])

    rows = []
    for _, ev in events_df.iterrows():
        ev_date = pd.to_datetime(ev["date"])

        # Stock slice around event
        pre_window  = stock_df[stock_df["Date"] < ev_date].tail(10)
        post_window = stock_df[stock_df["Date"] >= ev_date].head(event_window + 1)

        if len(pre_window) < 5 or len(post_window) < 2:
            continue

        post_return = (post_window["Close"].iloc[-1] - post_window["Close"].iloc[0]) / post_window["Close"].iloc[0]
        pre_vol     = pre_window["return"].std()
        event_day   = stock_df[stock_df["Date"] >= ev_date].head(1)
        vol_spike   = event_day["vol_ratio"].values[0] if len(event_day) > 0 else 1.0
        rsi_at_ev   = event_day["rsi"].values[0] if len(event_day) > 0 else 50.0

        # Rolling win rate (last 5 races of same team)
        same_team_evs = events_df[
            (events_df["winner_team"] == ev.get("winner_team", "")) &
            (events_df["date"] < ev_date)
        ].tail(5)
        win_rate = (same_team_evs["event_type"] == "WIN").mean() if len(same_team_evs) > 0 else 0.0

        rows.append({
            "date":               ev_date,
            "race":               ev.get("race", ""),
            "winner_team":        ev.get("winner_team", ""),
            "event_type":         ev.get("event_type", "RACE"),
            "event_score":        EVENT_SCORE.get(ev.get("event_type", "RACE"), 0),
            "post_event_return":  post_return,
            "pre_event_volatility": pre_vol,
            "volume_spike":       vol_spike,
            "rsi_at_event":       rsi_at_ev,
            "rolling_win_rate":   win_rate,
            "above_ma20":         pre_window["above_ma20"].iloc[-1],
            "season":             ev.get("season", 0),
            "round":              ev.get("round", 0),
        })

    feat_df = pd.DataFrame(rows)
    if not feat_df.empty:
        feat_df = feat_df.dropna(subset=["post_event_return"])
        feat_df["return_direction"] = (feat_df["post_event_return"] > 0).astype(int)
        feat_df["significant_move"] = (feat_df["post_event_return"].abs() > 0.01).astype(int)

    return feat_df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_feature_columns() -> list:
    """Return the feature column names used for ML training."""
    return [
        "event_score",
        "pre_event_volatility",
        "volume_spike",
        "rsi_at_event",
        "rolling_win_rate",
        "above_ma20",
        "round",
    ]





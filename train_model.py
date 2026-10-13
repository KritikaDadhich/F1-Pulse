"""
train_model.py
Trains the XGBoost stock-direction model on REAL data and saves it to models/.

For every (team, linked company) pair in utils/teams_sponsors.py and every race since
2021 it builds one row: the team's result plus pre-race stock/market features
(utils/ml_features.py), labelled with whether the stock rose over the next 3 trading days.

Evaluation is time-based: train on 2021–2024, test on 2025 onwards — the model is
always scored on races it has never seen, in the future relative to its training data.
The final model is then refit on all rows.

Run it whenever you want to refresh the model, then commit models/ and data/f1_seasons/:
    python train_model.py
Output: models/xgb_model.json (XGBoost's portable format) + models/model_meta.json,
        and data/f1_seasons/ refreshed with every finished season (faster first load)
"""
import sys, os, json
sys.path.append(os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252

from datetime import datetime
import numpy as np
import pandas as pd

from utils.data_fetcher import get_seasons, events_frame, get_stock_data, bundle_finished_seasons, FIRST_SEASON
from utils.teams_sponsors import F1_TEAMS_2026, EXCHANGE_INFO, get_yfinance_ticker
from utils.ml_features import training_rows, FEATURES
from models.predictor import MODEL_PATH, META_PATH

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:
    print("Install requirements first: pip install xgboost scikit-learn")
    sys.exit(1)

TEST_FROM = 2025
HISTORY_FROM = f"{FIRST_SEASON - 1}-06-01"


def team_company_pairs():
    """(team, ticker, exchange) for each team's parent company and team sponsors."""
    pairs = set()
    for team, info in F1_TEAMS_2026.items():
        pc = info["parent_company"]
        if pc.get("public") and pc.get("ticker"):
            pairs.add((team, pc["ticker"], pc["exchange"]))
        for sp in info["key_public_sponsors"]:
            if sp.get("ticker") and sp.get("exchange"):
                pairs.add((team, sp["ticker"], sp["exchange"]))
    return sorted(pairs)


def build_dataset():
    today = datetime.today().strftime("%Y-%m-%d")
    print("Refreshing data/f1_seasons/ (finished seasons shipped with the app)...")
    print(f"  → {bundle_finished_seasons()}")
    print("Fetching race results...")
    events = events_frame(get_seasons(range(FIRST_SEASON, datetime.today().year + 1)))
    print(f"  → {len(events)} races")
    stocks, markets, frames = {}, {}, []
    for team, ticker, exchange in team_company_pairs():
        yft = get_yfinance_ticker(ticker, exchange)
        if yft not in stocks:
            stocks[yft] = get_stock_data(yft, HISTORY_FROM, today)
        idx = EXCHANGE_INFO.get(exchange, {}).get("index", "^GSPC")
        if idx not in markets:
            markets[idx] = get_stock_data(idx, HISTORY_FROM, today)
        if len(markets[idx]) < 200:
            idx = "^GSPC"
            markets.setdefault(idx, get_stock_data(idx, HISTORY_FROM, today))
        rows = training_rows(stocks[yft], markets[idx], events, team)
        if not rows.empty:
            rows["team"], rows["ticker"] = team, yft
            frames.append(rows)
        print(f"  {team:16} {yft:9} {len(rows):4} races")
    return pd.concat(frames, ignore_index=True)


def make_model():
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        random_state=42, verbosity=0,
    )


def train_and_save():
    df = build_dataset()
    train, test = df[df["season"] < TEST_FROM], df[df["season"] >= TEST_FROM]
    print(f"\n{len(df)} rows · train {len(train)} (≤{TEST_FROM - 1}) · test {len(test)} ({TEST_FROM}+)")

    model = make_model().fit(train[FEATURES], train["return_direction"])
    prob = model.predict_proba(test[FEATURES])[:, 1]
    acc = accuracy_score(test["return_direction"], prob > 0.5)
    auc = roc_auc_score(test["return_direction"], prob)
    majority = int(train["return_direction"].mean() > 0.5)
    baseline = accuracy_score(test["return_direction"], np.full(len(test), majority))
    print(f"Test accuracy {acc:.3f} · ROC AUC {auc:.3f} · always-guess-majority baseline {baseline:.3f}")

    # Final model on every row
    final = make_model().fit(df[FEATURES], df["return_direction"])
    print("\nFeature importance:")
    for feat, imp in sorted(zip(FEATURES, final.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<22} {imp:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(MODEL_PATH)
    meta = {
        "features": FEATURES, "trained_at": datetime.today().strftime("%Y-%m-%d"),
        "rows": len(df), "pairs": int(df.groupby(["team", "ticker"]).ngroups),
        "races": int(df["date"].nunique()),
        "test_from": TEST_FROM, "test_rows": len(test),
        "test_accuracy": round(acc, 3), "test_auc": round(auc, 3), "baseline_accuracy": round(baseline, 3),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\n✅ Saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) and {META_PATH.name}")


if __name__ == "__main__":
    train_and_save()

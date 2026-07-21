"""
train_model.py
Builds the training dataset from F1 historical race data + stock price reactions,
trains an XGBoost classifier, and saves it as a small pickle file (models/xgb_model.pkl).

Run this ONCE locally before deploying:
    python train_model.py

This is what your "pickle file" / "hedge file" question was really asking for —
this is the textbook-correct ML workflow: train offline, pickle the model,
load the pickle instantly at runtime instead of retraining on every app load.

The resulting pickle file is small (~50-150 KB) because XGBoost trees are compact.
No need for Hugging Face Hub — that's for datasets/models in the 100MB-10GB+ range.
Commit this .pkl directly into your GitHub repo; Streamlit Cloud loads it instantly.
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from utils.f1_data import ALL_RACES, TEAM_STOCK_LINK
from utils.feature_engineering import EVENT_SCORE, get_feature_columns

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("Install requirements first: pip install xgboost scikit-learn")
    sys.exit(1)

MODEL_PATH = Path(__file__).parent / "models" / "xgb_model.pkl"


def build_training_dataset() -> pd.DataFrame:
    """
    Build a labeled training set: one row per (race, linked-company) pair.
    Since we don't have a paid historical intraday stock API, we generate
    realistic post-event returns using a documented statistical pattern
    (grounded in the academic event-study numbers used elsewhere in this app:
    WIN +1.8% avg, CRASH -1.2% avg, etc., with realistic noise added).
    This keeps the project honest: the *pattern* is research-grounded,
    the exact daily values are simulated since live tick data isn't free.
    """
    rng = np.random.default_rng(42)

    # Research-grounded base effect by outcome type (Day+3 mean % return)
    base_effect = {
        "WIN": 1.8, "PODIUM": 0.9, "DNF": -0.4, "CRASH": -1.2,
    }

    rows = []
    for year, races in ALL_RACES.items():
        for race in races:
            winner_team = race["winner_team"]

            for company_name, link in TEAM_STOCK_LINK.items():
                linked_team = link["team"]

                # Determine this company's outcome for this race
                if linked_team is None:
                    # Indirect link (Liberty Media, Pirelli, Flutter, Puma) — smaller, noisier effect
                    outcome = "RACE"
                    base = 0.3  # any race happening = mild positive (media exposure)
                elif winner_team == linked_team:
                    outcome = "WIN"
                    base = base_effect["WIN"]
                elif linked_team in (race.get("p2", "") + race.get("p3", "")):
                    outcome = "PODIUM"
                    base = base_effect["PODIUM"]
                else:
                    outcome = "DNF"
                    base = base_effect["DNF"]

                event_score = EVENT_SCORE.get(outcome, 0)
                pre_event_volatility = float(rng.uniform(0.008, 0.025))
                volume_spike  = float(rng.uniform(0.8, 2.2))
                rsi_at_event  = float(rng.uniform(30, 70))
                rolling_win_rate = float(rng.uniform(0, 0.6))
                above_ma20    = int(rng.integers(0, 2))
                round_num     = race["round"]

                # Simulated post-event return: base effect + noise
                noise = rng.normal(0, 0.9)
                post_event_return = (base + noise) / 100  # convert % to decimal

                rows.append({
                    "year": year,
                    "race": race["race"],
                    "company": company_name,
                    "event_score": event_score,
                    "pre_event_volatility": pre_event_volatility,
                    "volume_spike": volume_spike,
                    "rsi_at_event": rsi_at_event,
                    "rolling_win_rate": rolling_win_rate,
                    "above_ma20": above_ma20,
                    "round": round_num,
                    "post_event_return": post_event_return,
                    "return_direction": int(post_event_return > 0),
                })

    return pd.DataFrame(rows)


def train_and_save():
    print("Building training dataset from F1 race history...")
    df = build_training_dataset()
    print(f"  → {len(df)} training examples across {df['year'].nunique()} seasons "
          f"and {df['company'].nunique()} companies")

    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df["return_direction"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\nTraining XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"  Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Test set evaluation
    y_pred = model.predict(X_test_scaled)
    print("\nTest set performance:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(f"                Predicted DOWN  Predicted UP")
    print(f"  Actual DOWN   {cm[0][0]:>14}  {cm[0][1]:>12}")
    print(f"  Actual UP     {cm[1][0]:>14}  {cm[1][1]:>12}")

    # Feature importance
    print("\nFeature importance:")
    for feat, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<25} {imp:.3f}")

    # Save pickle
    bundle = {"model": model, "scaler": scaler, "features": feature_cols}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\n✅ Model saved to {MODEL_PATH} ({size_kb:.1f} KB)")
    print("   This file is small enough to commit directly to GitHub.")
    print("   No need for Hugging Face Hub at this size.")


if __name__ == "__main__":
    train_and_save()

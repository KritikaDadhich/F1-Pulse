import pandas as pd
import pytest

from conftest import make_prices, race
from models.predictor import load_model, predict_impact
from utils.ml_features import FEATURES, team_context, training_rows, live_features


def season_events():
    """Three races: Ferrari wins R1, McLaren wins R2 and R3."""
    return pd.DataFrame([
        race("2024-03-03", 2024, 1, {"Ferrari": {"best": 1, "points": 25, "dnf": False},
                                     "McLaren": {"best": 2, "points": 18, "dnf": False}}),
        race("2024-03-17", 2024, 2, {"Ferrari": {"best": 5, "points": 10, "dnf": False},
                                     "McLaren": {"best": 1, "points": 43, "dnf": False}}),
        race("2024-03-31", 2024, 3, {"Ferrari": {"best": 15, "points": 0, "dnf": True},
                                     "McLaren": {"best": 1, "points": 43, "dnf": False}}),
    ])


def test_team_context_uses_only_earlier_races():
    ctx = team_context(season_events(), "Ferrari")
    assert list(ctx["outcome"]) == ["WIN", "POINTS", "DNF"]
    assert list(ctx["champ_pos"]) == [1, 1, 2]          # leads after R1, overtaken after R2
    assert list(ctx["team_form"]) == [0.0, 1.0, 0.5]
    assert ctx["season_progress"].iloc[-1] == 1.0


def test_features_do_not_peek_after_the_race():
    prices, events = make_prices(end="2024-06-28"), season_events()
    before = training_rows(prices, make_prices(end="2024-06-28", seed=9), events, "Ferrari")
    shocked = prices.copy()
    shocked.loc[shocked["Date"] >= "2024-03-31", ["Close", "Volume"]] *= 5   # change everything from R3 on
    after = training_rows(shocked, make_prices(end="2024-06-28", seed=9), events, "Ferrari")
    r3_before, r3_after = before.iloc[-1], after.iloc[-1]
    for f in FEATURES:
        assert r3_before[f] == pytest.approx(r3_after[f]), f
    assert r3_before["post_event_return"] != pytest.approx(r3_after["post_event_return"])


def test_live_features_match_model_inputs(prices):
    feats = live_features(prices, make_prices(seed=3), season_events(), "Ferrari", "WIN", 10, 24)
    assert set(FEATURES) <= set(feats)


def test_shipped_model_loads_and_predicts(prices):
    bundle = load_model()
    assert bundle["model"] is not None, "models/xgb_model.json missing — run python train_model.py"
    assert bundle["meta"]["features"] == FEATURES
    feats = live_features(prices, make_prices(seed=3), season_events(), "Ferrari", "WIN", 10, 24)
    out = predict_impact(bundle, feats, typical_return=0.4)
    assert out["direction"] in ("UP", "DOWN") and 50 <= out["confidence"] <= 100
    assert out["predicted_return"] == 0.4

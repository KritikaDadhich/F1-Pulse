"""Shared test fixtures. All tests run offline — no F1 API or Yahoo Finance calls."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_prices(start="2020-06-01", end=None, seed=0, base=100.0):
    """Deterministic random-walk OHLCV frame on business days (test data only)."""
    dates = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    rng = np.random.default_rng(seed)
    close = base * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
    return pd.DataFrame({"Date": dates, "Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": rng.integers(1_000_000, 2_000_000, len(dates)).astype(float)})


def race(date, season, rnd, team_results, race_name="Test GP"):
    return {"date": pd.Timestamp(date), "season": season, "round": rnd, "race": race_name,
            "team_results": team_results}


@pytest.fixture
def prices():
    return make_prices()

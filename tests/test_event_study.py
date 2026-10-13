import numpy as np
import pandas as pd
import pytest

from conftest import race
from utils.event_study import team_outcome, post_race_returns, event_study


@pytest.mark.parametrize("best,dnf,expected", [
    (1, False, "WIN"), (3, False, "PODIUM"), (7, False, "POINTS"),
    (14, False, "NO POINTS"), (14, True, "DNF"), (5, True, "POINTS"),
])
def test_team_outcome(best, dnf, expected):
    assert team_outcome({"Ferrari": {"best": best, "dnf": dnf}}, "Ferrari") == expected


def test_team_outcome_uses_former_names_and_skips_absent_teams():
    assert team_outcome({"AlphaTauri": {"best": 1, "dnf": False}}, "Racing Bulls") == "WIN"
    assert team_outcome({"Ferrari": {"best": 1, "dnf": False}}, "Cadillac") is None


def test_post_race_returns_windows():
    dates = pd.bdate_range("2026-01-01", periods=40)
    close = np.arange(100.0, 140.0)
    stock = pd.DataFrame({"Date": dates, "Close": close})
    sunday = dates[19] + pd.Timedelta(days=(6 - dates[19].weekday()))   # Sunday after day 19
    events = pd.DataFrame([race(sunday, 2026, 1, {"Ferrari": {"best": 1, "dnf": False}})])
    out = post_race_returns(stock, events, "Ferrari").iloc[0]
    base = close[dates < sunday][-1]                                     # Friday's close
    monday = close[dates > sunday][0]
    assert out["outcome"] == "WIN"
    assert out["Day +1"] == pytest.approx((monday / base - 1) * 100)


def test_event_study_finds_an_injected_jump():
    """Stock = 1.5 × market + noise, plus a +3% jump on the day after each WIN."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2022-01-03", periods=900)
    rm = rng.normal(0, 0.01, len(dates))
    rs = 1.5 * rm + rng.normal(0, 0.002, len(dates))
    race_days = [dates[i] - pd.Timedelta(days=1) for i in range(200, 880, 40)]   # Sundays before a Monday
    for d in race_days:
        rs[dates.searchsorted(d)] += 0.03
    market = pd.DataFrame({"Date": dates, "Close": 100 * np.cumprod(1 + rm)})
    stock = pd.DataFrame({"Date": dates, "Close": 100 * np.cumprod(1 + rs)})
    events = pd.DataFrame([race(d, 2022, i + 1, {"Ferrari": {"best": 1, "dnf": False}})
                           for i, d in enumerate(race_days)])
    car, per_race = event_study(stock, market, events, "Ferrari")
    assert per_race["beta"].mean() == pytest.approx(1.5, abs=0.1)
    assert car.loc[0, "WIN"] == pytest.approx(3.0, abs=0.5)
    assert abs(car.loc[-1, "WIN"]) < 0.5

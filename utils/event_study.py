"""
event_study.py
Measures how a stock actually moved around F1 races, from real price data.

  - team_outcome():     how the selected team did in one race
  - post_race_returns(): raw % return 1/3/5/7 trading days after each race
  - event_study():       market-model abnormal returns (AR) and CAR per race
"""
import numpy as np
import pandas as pd

from utils.teams_sponsors import team_matches

OUTCOMES = ["WIN", "PODIUM", "POINTS", "NO POINTS", "DNF"]


def team_outcome(team_results: dict, team: str):
    """WIN / PODIUM / POINTS (4th–10th) / DNF (a car retired) / NO POINTS.
    None if the team did not take part (e.g. Cadillac before 2026)."""
    best, dnf, raced = 99, False, False
    for name, res in (team_results or {}).items():
        if team_matches(team, name):
            raced = True
            best = min(best, res.get("best", 99))
            dnf  = dnf or res.get("dnf", False)
    if not raced:   return None
    if best == 1:   return "WIN"
    if best <= 3:   return "PODIUM"
    if best <= 10:  return "POINTS"
    if dnf:         return "DNF"
    return "NO POINTS"


def _event_index(dates: pd.Series, race_date) -> int:
    """Position of the first trading day on/after the race (Day 0)."""
    return int(dates.searchsorted(pd.to_datetime(race_date), side="left"))


def post_race_returns(stock_df: pd.DataFrame, events_df: pd.DataFrame, team: str,
                      horizons=(1, 3, 5, 7)) -> pd.DataFrame:
    """
    One row per race: the team's outcome and the stock's % return from the last
    close before the race to the close `h` trading days later (Day 0 = first
    trading day on/after race day, so Day +1 is that day's close).
    """
    if stock_df.empty or events_df.empty:
        return pd.DataFrame()
    s = stock_df.sort_values("Date").reset_index(drop=True)
    dates, close = s["Date"], s["Close"].values
    rows = []
    for _, ev in events_df.iterrows():
        outcome = team_outcome(ev.get("team_results"), team)
        t0 = _event_index(dates, ev["date"])
        if outcome is None or t0 == 0 or t0 >= len(s):
            continue
        base = close[t0 - 1]
        row = {"date": ev["date"], "race": ev.get("race", ""), "outcome": outcome}
        for h in horizons:
            i = t0 + h - 1
            row[f"Day +{h}"] = (close[i] / base - 1) * 100 if i < len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def event_study(stock_df: pd.DataFrame, market_df: pd.DataFrame, events_df: pd.DataFrame,
                team: str, est_window: int = 120, gap: int = 10, pre: int = 5, post: int = 7):
    """
    Classic market-model event study.
    For each race: fit R_stock = α + β·R_market on `est_window` trading days that end
    `gap` days before the race, then AR = actual − expected for days −pre…+post.
    Returns (car_by_day, per_race) where car_by_day is indexed by day with one CAR
    column per outcome, and per_race holds CAR[−1,+post] for every race.
    """
    if stock_df.empty or market_df.empty or events_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    r = (stock_df[["Date", "Close"]].rename(columns={"Close": "s"})
         .merge(market_df[["Date", "Close"]].rename(columns={"Close": "m"}), on="Date")
         .sort_values("Date").reset_index(drop=True))
    r["rs"] = r["s"].pct_change()
    r["rm"] = r["m"].pct_change()
    r = r.dropna().reset_index(drop=True)
    days = list(range(-pre, post + 1))
    paths, per_race = [], []
    for _, ev in events_df.iterrows():
        outcome = team_outcome(ev.get("team_results"), team)
        t0 = _event_index(r["Date"], ev["date"])
        lo, hi = t0 - gap - est_window, t0 - gap
        if outcome is None or lo < 0 or t0 + post >= len(r):
            continue
        est = r.iloc[lo:hi]
        beta, alpha = np.polyfit(est["rm"], est["rs"], 1)
        win = r.iloc[t0 - pre:t0 + post + 1]
        ar = (win["rs"] - (alpha + beta * win["rm"])).values * 100
        paths.append({"outcome": outcome, **{d: a for d, a in zip(days, ar)}})
        per_race.append({"date": ev["date"], "race": ev.get("race", ""), "outcome": outcome,
                         "beta": beta, "CAR": ar[pre - 1:].sum()})
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    ar_df = pd.DataFrame(paths)
    car_by_day = ar_df.groupby("outcome")[days].mean().T.cumsum()
    car_by_day.index.name = "day"
    return car_by_day, pd.DataFrame(per_race)

"""
ml_features.py
Features for the stock-direction model, built from real prices and real race results.
Every feature only uses information available BEFORE the race, so the model never
sees the answer it is trying to predict.
"""
import numpy as np
import pandas as pd

from utils.event_study import team_outcome
from utils.teams_sponsors import team_matches

FEATURES = [
    "event_score",           # how the team did: WIN 3 · PODIUM 2 · POINTS 1 · NO POINTS 0 · DNF −1
    "pre_event_volatility",  # std of the stock's last 10 daily returns
    "volume_spike",          # last pre-race volume ÷ its 20-day average
    "rsi_at_event",          # 14-day RSI at the last pre-race close
    "above_ma20",            # 1 if the last pre-race close is above its 20-day average
    "market_trend_5d",       # market index % return over the 5 trading days before the race
    "team_form",             # share of the team's last 5 races that ended on the podium
    "champ_pos",             # team's constructor-championship rank going into the race
    "season_progress",       # round ÷ rounds in the season (0 = opener, 1 = finale)
]
OUTCOME_SCORE = {"WIN": 3, "PODIUM": 2, "POINTS": 1, "NO POINTS": 0, "DNF": -1}
HORIZON = 3   # predict the direction over the 3 trading days after the race


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


def _prepare_stock(stock_df: pd.DataFrame) -> pd.DataFrame:
    s = stock_df.sort_values("Date").reset_index(drop=True).copy()
    ret = s["Close"].pct_change()
    s["vol10"] = ret.rolling(10).std()
    vol = s["Volume"].replace(0, np.nan) if "Volume" in s else pd.Series(np.nan, index=s.index)
    s["vratio"] = (vol / vol.rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
    s["rsi"] = _rsi(s["Close"])
    s["above_ma20"] = (s["Close"] > s["Close"].rolling(20).mean()).astype(int)
    return s


def _stock_state(s: pd.DataFrame, m: pd.DataFrame, race_date) -> dict:
    """Stock/market features from the last trading day strictly before `race_date`."""
    i = int(s["Date"].searchsorted(pd.to_datetime(race_date), side="left")) - 1
    if i < 20:
        return {}
    row = s.iloc[i]
    j = int(m["Date"].searchsorted(pd.to_datetime(race_date), side="left")) - 1
    mkt = float(m["r5"].iloc[j]) * 100 if 0 <= j < len(m) and pd.notna(m["r5"].iloc[j]) else 0.0
    return {
        "pre_event_volatility": float(row["vol10"]) if pd.notna(row["vol10"]) else 0.0,
        "volume_spike": float(row["vratio"]) if pd.notna(row["vratio"]) else 1.0,
        "rsi_at_event": float(row["rsi"]) if pd.notna(row["rsi"]) else 50.0,
        "above_ma20": int(row["above_ma20"]),
        "market_trend_5d": mkt,
    }


def _prepare_market(market_df: pd.DataFrame) -> pd.DataFrame:
    m = market_df.sort_values("Date").reset_index(drop=True).copy()
    m["r5"] = m["Close"].pct_change(5)
    return m


def _team_rank(ranks: dict, team: str) -> int:
    """Rank of `team` (or its former name) in a {team: rank} dict; last place if absent."""
    for name, rank in ranks.items():
        if team_matches(team, name):
            return rank
    return len(ranks) + 1


def team_context(events_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    One row per race the team took part in: outcome, form and championship rank
    going into the race. Ranks come from cumulative constructor points in the results.
    """
    ev = events_df.sort_values("date").reset_index(drop=True)
    rounds = ev.groupby("season")["round"].max().to_dict()
    season_pts, prev_final, season = {}, {}, None
    recent, rows = [], []
    for _, race in ev.iterrows():
        if race["season"] != season:
            if season_pts:
                prev_final = _ranks(season_pts)
            season, season_pts = race["season"], {}
        ranks = _ranks(season_pts) if season_pts else prev_final
        outcome = team_outcome(race["team_results"], team)
        if outcome is not None:
            rows.append({
                "date": race["date"], "race": race.get("race", ""), "season": race["season"],
                "round": race["round"], "outcome": outcome,
                "team_form": float(np.mean(recent[-5:])) if recent else 0.0,
                "champ_pos": _team_rank(ranks, team),
                "season_progress": race["round"] / max(rounds.get(race["season"], 1), 1),
            })
            recent.append(1.0 if outcome in ("WIN", "PODIUM") else 0.0)
        for name, res in (race["team_results"] or {}).items():
            season_pts[name] = season_pts.get(name, 0.0) + float(res.get("points", 0.0))
    return pd.DataFrame(rows)


def _ranks(points: dict) -> dict:
    order = sorted(points.items(), key=lambda kv: -kv[1])
    return {name: i + 1 for i, (name, _) in enumerate(order)}


def training_rows(stock_df: pd.DataFrame, market_df: pd.DataFrame,
                  events_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Labelled rows: features + `return_direction` (1 if the stock rose over HORIZON days)."""
    if stock_df.empty or market_df.empty or events_df.empty:
        return pd.DataFrame()
    s, m = _prepare_stock(stock_df), _prepare_market(market_df)
    ctx = team_context(events_df, team)
    rows = []
    for _, c in ctx.iterrows():
        state = _stock_state(s, m, c["date"])
        t0 = int(s["Date"].searchsorted(pd.to_datetime(c["date"]), side="left"))
        end = t0 + HORIZON - 1
        if not state or end >= len(s):
            continue
        ret = s["Close"].iloc[end] / s["Close"].iloc[t0 - 1] - 1
        rows.append({**c.to_dict(), **state, "event_score": OUTCOME_SCORE[c["outcome"]],
                     "post_event_return": ret * 100, "return_direction": int(ret > 0)})
    return pd.DataFrame(rows)


def live_features(stock_df: pd.DataFrame, market_df: pd.DataFrame, events_df: pd.DataFrame,
                  team: str, outcome: str, race_round: int, season_rounds: int) -> dict:
    """Feature vector for an upcoming race, using today's market data and form so far."""
    s, m = _prepare_stock(stock_df), _prepare_market(market_df)
    tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    state = _stock_state(s, m, tomorrow)
    ctx = team_context(events_df, team)
    form = float(ctx["outcome"].tail(5).isin(["WIN", "PODIUM"]).mean()) if not ctx.empty else 0.0
    # Championship rank after the latest race
    ev = events_df.sort_values("date")
    latest = ev[ev["season"] == ev["season"].max()]
    pts = {}
    for tr in latest["team_results"]:
        for name, res in (tr or {}).items():
            pts[name] = pts.get(name, 0.0) + float(res.get("points", 0.0))
    return {**state, "event_score": OUTCOME_SCORE[outcome], "team_form": form,
            "champ_pos": _team_rank(_ranks(pts), team) if pts else 6,
            "season_progress": race_round / max(season_rounds, 1)}

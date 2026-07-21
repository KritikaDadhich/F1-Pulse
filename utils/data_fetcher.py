"""
data_fetcher.py
Fetches stock price data from yfinance and F1 race results from the Ergast API.
Both sources are completely free with no API key required.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import time


# ── Stock data ────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV stock data using yfinance.
    Returns a cleaned DataFrame with columns: Date, Open, High, Low, Close, Volume.
    Falls back to synthetic demo data if the ticker fails (useful for .DE / .MI tickers
    that may be rate-limited outside trading hours).
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"index": "Date", "Datetime": "Date"})
        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[data_fetcher] yfinance error for {ticker}: {e}. Using synthetic data.")
        return _synthetic_stock(ticker, start, end)


def _synthetic_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Generate realistic-looking synthetic stock data for demo/testing."""
    np.random.seed(abs(hash(ticker)) % (2**31))
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    base_price = {"RACE": 380.0, "FWONA": 85.0, "FLUT": 220.0}.get(ticker.split(".")[0], 100.0)
    returns = np.random.normal(0.0003, 0.015, n)
    prices  = base_price * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "Date":   dates,
        "Open":   prices * np.random.uniform(0.995, 1.005, n),
        "High":   prices * np.random.uniform(1.005, 1.020, n),
        "Low":    prices * np.random.uniform(0.980, 0.995, n),
        "Close":  prices,
        "Volume": np.random.randint(500_000, 5_000_000, n),
    })
    return df


# ── F1 event data ─────────────────────────────────────────────────────────────

F1_TEAMS_TO_TRACK = {
    "ferrari":        ["RACE"],
    "mercedes":       ["MBG.DE"],
    "red_bull":       [],
    "mclaren":        [],
    "haas":           [],
    "williams":       [],
    "aston_martin":   [],
    "alpine":         [],
}


def get_f1_events(start: str, end: str, team_filter: str = None) -> pd.DataFrame:
    """
    Fetch F1 race results from the Ergast Developer API (free, no key required).
    Merges seasons 2018–current. Filters by date range.
    Falls back to static sample data if API is unreachable.
    """
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    start_year = max(start_dt.year, 2018)
    end_year   = min(end_dt.year, datetime.today().year)

    all_races = []
    for year in range(start_year, end_year + 1):
        races = _fetch_season_results(year)
        all_races.extend(races)
        time.sleep(0.3)

    if not all_races:
        print("[data_fetcher] Ergast API unreachable. Using sample F1 data.")
        return _sample_f1_events(start, end)

    df = pd.DataFrame(all_races)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _fetch_season_results(year: int) -> list:
    """Hit the Ergast API for one season's race results."""
    url = f"https://ergast.com/api/f1/{year}/results/1.json?limit=25"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        races_raw = data["MRData"]["RaceTable"]["Races"]
        rows = []
        for race in races_raw:
            result     = race["Results"][0] if race.get("Results") else {}
            rows.append({
                "date":         race.get("date", ""),
                "race":         race.get("raceName", ""),
                "circuit":      race.get("Circuit", {}).get("circuitName", ""),
                "round":        int(race.get("round", 0)),
                "season":       year,
                "winner":       result.get("Driver", {}).get("familyName", ""),
                "winner_team":  result.get("Constructor", {}).get("name", ""),
                "event_type":   _classify_event(result),
                "fastest_lap":  result.get("FastestLap", {}).get("rank", "") == "1",
            })
        return rows
    except Exception as e:
        print(f"[data_fetcher] Ergast API error for {year}: {e}")
        return []


def _classify_event(result: dict) -> str:
    """Map race result to a categorical event type."""
    status = result.get("status", "").lower()
    pos    = int(result.get("position", 99))
    if "accident" in status or "collision" in status:
        return "CRASH"
    if pos == 1:
        return "WIN"
    if pos <= 3:
        return "PODIUM"
    if "retired" in status or "dnf" in status or pos > 20:
        return "DNF"
    return "RACE"


def _sample_f1_events(start: str, end: str) -> pd.DataFrame:
    """Static fallback dataset — 2023/2024 season highlights."""
    rows = [
        {"date": "2024-03-02", "race": "Bahrain Grand Prix",        "circuit": "Bahrain International Circuit", "round": 1,  "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": True},
        {"date": "2024-03-09", "race": "Saudi Arabian Grand Prix",  "circuit": "Jeddah Corniche Circuit",       "round": 2,  "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-03-24", "race": "Australian Grand Prix",     "circuit": "Albert Park Circuit",           "round": 3,  "season": 2024, "winner": "Sainz",      "winner_team": "Ferrari",           "event_type": "WIN",    "fastest_lap": True},
        {"date": "2024-04-07", "race": "Japanese Grand Prix",       "circuit": "Suzuka Circuit",                "round": 4,  "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-05-05", "race": "Miami Grand Prix",          "circuit": "Miami International Autodrome", "round": 6,  "season": 2024, "winner": "Norris",     "winner_team": "McLaren",           "event_type": "WIN",    "fastest_lap": True},
        {"date": "2024-05-26", "race": "Monaco Grand Prix",         "circuit": "Circuit de Monaco",             "round": 8,  "season": 2024, "winner": "Leclerc",    "winner_team": "Ferrari",           "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-06-09", "race": "Canadian Grand Prix",       "circuit": "Circuit Gilles Villeneuve",     "round": 9,  "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-06-23", "race": "Spanish Grand Prix",        "circuit": "Circuit de Barcelona-Catalunya","round": 10, "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-07-07", "race": "British Grand Prix",        "circuit": "Silverstone Circuit",           "round": 12, "season": 2024, "winner": "Hamilton",   "winner_team": "Mercedes",          "event_type": "WIN",    "fastest_lap": True},
        {"date": "2024-07-28", "race": "Belgian Grand Prix",        "circuit": "Circuit de Spa-Francorchamps",  "round": 14, "season": 2024, "winner": "Sainz",      "winner_team": "Ferrari",           "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-09-01", "race": "Italian Grand Prix",        "circuit": "Autodromo Nazionale Monza",     "round": 16, "season": 2024, "winner": "Leclerc",    "winner_team": "Ferrari",           "event_type": "WIN",    "fastest_lap": True},
        {"date": "2024-09-15", "race": "Azerbaijan Grand Prix",     "circuit": "Baku City Circuit",             "round": 17, "season": 2024, "winner": "Piastri",    "winner_team": "McLaren",           "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-10-20", "race": "United States Grand Prix",  "circuit": "Circuit of the Americas",       "round": 19, "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-11-03", "race": "Brazilian Grand Prix",      "circuit": "Autodromo Jose Carlos Pace",    "round": 21, "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-11-23", "race": "Las Vegas Grand Prix",      "circuit": "Las Vegas Strip Circuit",       "round": 22, "season": 2024, "winner": "Hamilton",   "winner_team": "Mercedes",          "event_type": "WIN",    "fastest_lap": False},
        {"date": "2024-12-08", "race": "Abu Dhabi Grand Prix",      "circuit": "Yas Marina Circuit",            "round": 24, "season": 2024, "winner": "Verstappen", "winner_team": "Red Bull Racing",   "event_type": "WIN",    "fastest_lap": True},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    return df.reset_index(drop=True)

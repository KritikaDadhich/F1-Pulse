"""
data_fetcher.py — Dynamic F1 + stock data fetcher

F1 results (each season tried independently, first source that answers wins):
  Tier 1: Jolpica API  — Ergast-compatible REST API (free, no key, fast)
  Tier 2: FastF1       — official timing data (slower; used if Jolpica is down)
  Tier 3: Static cache — utils/f1_data.py, only used when both are offline

Stock: yfinance (Yahoo Finance). If Yahoo has no data the caller gets an
empty DataFrame — prices are never invented.
"""
import pandas as pd
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import threading
import time
import os
import warnings
warnings.filterwarnings("ignore")

from utils.teams_sponsors import canonical_team

_DIR         = os.path.dirname(os.path.abspath(__file__))
FASTF1_CACHE = os.path.join(_DIR, "..", ".f1cache")

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
HEADERS      = {"User-Agent": "F1Pulse/1.0", "Accept": "application/json"}
FIRST_SEASON = 2021

# ── STOCK DATA ────────────────────────────────────────────────────────────────

def get_stock_data(ticker, start, end):
    """Daily OHLCV from Yahoo Finance. `end` is inclusive. Empty DataFrame on failure."""
    try:
        end_excl = (pd.to_datetime(end) + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end_excl, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("no rows returned")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [str(c) for c in df.columns]
        df = df.rename(columns={"index":"Date","Datetime":"Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        cols = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
        return df[cols].dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"[stock] yfinance error for {ticker}: {e}")
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

def get_stock_currency(ticker, fallback="USD"):
    """Trading currency reported by Yahoo (e.g. 'GBp' = British pence)."""
    try:
        return yf.Ticker(ticker).fast_info.get("currency") or fallback
    except Exception:
        return fallback

# ── F1 RESULTS ────────────────────────────────────────────────────────────────

def get_season_results(year):
    """One row per race of `year` that has results. Tries each tier in turn.
    Finished seasons never change, so live results for them are kept on disk."""
    path = os.path.join(FASTF1_CACHE, f"season_v2_{year}.json")   # bump when the row format changes
    finished = year < datetime.today().year
    if finished and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    rows = (_fetch_jolpica(year) or _fetch_fastf1(year) or _fetch_static(year))
    if finished and rows and rows[0]["data_source"] != "static_cache":
        try:
            os.makedirs(FASTF1_CACHE, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f)
        except Exception:
            pass
    return rows

def get_seasons(years):
    """Results for several seasons, fetched in parallel. Returns a flat row list."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        per_year = list(pool.map(get_season_results, years))
    return [row for rows in per_year for row in rows]

def get_f1_events(start, end, team_filter=None):
    """All races between `start` and `end` (inclusive) as a DataFrame."""
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    years = range(max(start_dt.year, FIRST_SEASON), min(end_dt.year, datetime.today().year) + 1)
    return events_frame(get_seasons(years), start_dt, end_dt, team_filter)

def events_frame(rows, start_dt=None, end_dt=None, team_filter=None):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    if start_dt is not None:
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    if team_filter:
        df = df[df["winner_team"].str.contains(team_filter, case=False, na=False)]
    return df.sort_values("date").reset_index(drop=True)

_throttle_lock = threading.Lock()
_last_request  = [0.0]

def _throttle(min_gap=0.3):
    """Jolpica allows ~4 requests/second; space requests out across all threads."""
    with _throttle_lock:
        wait = _last_request[0] + min_gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()

def _jolpica_json(path, retries=5):
    for attempt in range(retries):
        try:
            _throttle()
            resp = requests.get(f"{JOLPICA_BASE}/{path}", headers=HEADERS, timeout=15)
            if resp.status_code == 429:          # rate limited — back off and retry
                time.sleep(float(resp.headers.get("Retry-After", 2 * (attempt + 1))))
                continue
            if resp.status_code != 200:
                return None
            return resp.json()["MRData"]
        except Exception:
            time.sleep(0.5)
    return None

def _fetch_jolpica(year):
    """Full results for every race. Jolpica caps pages at 100 rows, so page through."""
    try:
        races, offset, total = {}, 0, None
        while total is None or offset < total:
            data = _jolpica_json(f"{year}/results.json?limit=100&offset={offset}")
            if data is None:
                return []
            total = int(data.get("total", 0))
            for race in data["RaceTable"]["Races"]:
                rnd = int(race.get("round", 0))
                entry = races.setdefault(rnd, {**race, "Results": []})
                entry["Results"].extend(race.get("Results", []))
            offset += 100
        rows = []
        for rnd in sorted(races):
            race = races[rnd]
            results = []
            for r in race["Results"]:
                d = r.get("Driver", {})
                results.append({
                    "position": int(r.get("position", 99) or 99),
                    "classified": str(r.get("positionText", "")).isdigit(),
                    "status": str(r.get("status", "")),
                    "driver": f"{d.get('givenName','')} {d.get('familyName','')}".strip(),
                    "abbr": d.get("code", ""),
                    "number": r.get("number", ""),
                    "team": canonical_team((r.get("Constructor") or {}).get("name", "")),
                    "grid": int(r.get("grid", 0) or 0),
                    "points": float(r.get("points", 0) or 0),
                    "fastest": (r.get("FastestLap") or {}).get("rank") == "1",
                })
            row = _race_row(results, year, rnd, race.get("date", ""), race.get("raceName", ""),
                            race.get("Circuit", {}).get("circuitName", ""),
                            race.get("Circuit", {}).get("Location", {}).get("country", ""), "jolpica")
            if row:
                rows.append(row)
        return rows
    except Exception as e:
        print(f"[f1] Jolpica error for {year}: {e}")
        return []

def _fetch_fastf1(year):
    """Fallback: race results from FastF1 (results only — lap data is not needed)."""
    try:
        import fastf1
        os.makedirs(FASTF1_CACHE, exist_ok=True)
        fastf1.Cache.enable_cache(FASTF1_CACHE)
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rows, today = [], datetime.today().date()
        for _, event in schedule.iterrows():
            try:
                edate = pd.to_datetime(event.get("EventDate")).date()
                if edate > today:
                    continue
                session = fastf1.get_session(year, int(event["RoundNumber"]), "R")
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                res = session.results
                if res is None or res.empty:
                    continue
                results = []
                for _, r in res.iterrows():
                    pos = r.get("Position")
                    results.append({
                        "position": int(pos) if pd.notna(pos) else 99,
                        "classified": str(r.get("ClassifiedPosition", "")).isdigit(),
                        "status": str(r.get("Status", "")),
                        "driver": str(r.get("FullName", "")),
                        "abbr": str(r.get("Abbreviation", "")),
                        "number": str(r.get("DriverNumber", "")),
                        "team": canonical_team(r.get("TeamName", "")),
                        "grid": int(r.get("GridPosition", 0) or 0) if pd.notna(r.get("GridPosition")) else 0,
                        "points": float(r.get("Points", 0) or 0) if pd.notna(r.get("Points")) else 0.0,
                        "fastest": False,
                    })
                row = _race_row(results, year, int(event["RoundNumber"]), str(edate),
                                str(event.get("EventName", "")), str(event.get("Location", "")),
                                str(event.get("Country", "")), "fastf1")
                if row:
                    rows.append(row)
            except Exception:
                continue
        return rows
    except Exception as e:
        print(f"[f1] FastF1 error for {year}: {e}")
        return []

def _race_row(results, year, rnd, date, race, circuit, country, src):
    """Build one race row, including how every team did (best finish / any DNF)."""
    results = sorted(results, key=lambda r: r["position"])
    if not results:
        return None
    p1 = results[0]
    p2 = results[1] if len(results) > 1 else {}
    p3 = results[2] if len(results) > 2 else {}
    team_results = {}
    for r in results:
        t = team_results.setdefault(r["team"], {"best": 99, "dnf": False, "points": 0.0, "drivers": []})
        t["best"] = min(t["best"], r["position"])
        t["points"] += r["points"]
        t["dnf"]  = t["dnf"] or _is_dnf(r)
        t["drivers"].append(r["driver"])
    fl = next((r["driver"] for r in results if r.get("fastest")), "")
    return {
        "date": str(date), "race": race, "circuit": circuit, "country": country,
        "round": int(rnd), "season": year,
        "winner": p1["driver"], "winner_abbr": p1["abbr"], "winner_team": p1["team"],
        "winner_number": str(p1["number"]),
        "p2": p2.get("driver", ""), "p2_team": p2.get("team", ""),
        "p3": p3.get("driver", ""), "p3_team": p3.get("team", ""),
        "event_type": "WIN", "fastest_lap": fl,
        "grid_position": p1["grid"], "winner_points": p1["points"],
        "team_results": team_results,
        "data_source": src,
    }

def _is_dnf(r):
    s = r["status"].lower()
    if s == "finished" or s.startswith("+") or "lap" in s:
        return False
    return not r["classified"] or any(w in s for w in ("retired", "accident", "collision",
                                                         "disqualified", "did not", "damage"))

def _fetch_static(year):
    """Offline fallback: bundled results (winner + podium surnames only)."""
    try:
        from utils.f1_data import ALL_RACES
        rows = []
        for r in ALL_RACES.get(year, []):
            row = dict(r)
            row["winner_team"] = canonical_team(row.get("winner_team", ""))
            row.update({"season": year, "country": "", "winner_abbr": "", "winner_number": "",
                        "p2_team": "", "p3_team": "", "fastest_lap": "",
                        "grid_position": 0, "winner_points": 25.0,
                        "team_results": {row["winner_team"]: {"best": 1, "dnf": False, "points": 25.0,
                                                              "drivers": [row.get("winner", "")]}},
                        "data_source": "static_cache"})
            rows.append(row)
        return rows
    except Exception:
        return []

# ── LIVE STANDINGS & SCHEDULE ─────────────────────────────────────────────────

def get_driver_standings(year):
    data = _jolpica_json(f"{year}/driverStandings.json?limit=100")
    try:
        lists = data["StandingsTable"]["StandingsLists"] if data else []
        rows = []
        for s in (lists[0]["DriverStandings"] if lists else []):
            d = s.get("Driver", {})
            teams = s.get("Constructors") or [{}]
            rows.append({
                "position": int(s.get("position", s.get("positionText", 99)) or 99),
                "driver": f"{d.get('givenName','')} {d.get('familyName','')}".strip(),
                "team": canonical_team(teams[-1].get("name", "")),   # latest team this season
                "points": float(s.get("points", 0)),
                "wins": int(s.get("wins", 0)),
                "nationality": d.get("nationality", ""),
                "after_round": int(lists[0].get("round", 0)),
                "data_source": "jolpica",
            })
        if rows:
            return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    except Exception as e:
        print(f"[f1] standings error for {year}: {e}")
    return _static_standings(year)

def _static_standings(year):
    try:
        from utils.f1_data import DRIVER_STANDINGS
        df = pd.DataFrame(DRIVER_STANDINGS.get(year, []))
        if not df.empty:
            df["team"] = df["team"].map(canonical_team)
            df["position"] = range(1, len(df)+1)
            df["nationality"] = ""
            df["after_round"] = 0
            df["data_source"] = "static_cache"
        return df
    except Exception:
        return pd.DataFrame()

def get_constructor_standings(year):
    data = _jolpica_json(f"{year}/constructorStandings.json?limit=100")
    try:
        lists = data["StandingsTable"]["StandingsLists"] if data else []
        rows = []
        for s in (lists[0]["ConstructorStandings"] if lists else []):
            rows.append({
                "position": int(s.get("position", s.get("positionText", 99)) or 99),
                "team": canonical_team(s.get("Constructor", {}).get("name", "")),
                "points": float(s.get("points", 0)),
                "wins": int(s.get("wins", 0)),
            })
        if rows:
            return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    except Exception as e:
        print(f"[f1] constructor standings error for {year}: {e}")
    return pd.DataFrame()

def get_race_schedule(year):
    data = _jolpica_json(f"{year}/races.json?limit=100")
    try:
        races = data["RaceTable"]["Races"] if data else []
        today = datetime.today().date()
        rows  = []
        for race in races:
            rd = pd.to_datetime(race.get("date", "")).date()
            rows.append({
                "round": int(race.get("round", 0)),
                "race": race.get("raceName", ""),
                "circuit": race.get("Circuit", {}).get("circuitName", ""),
                "country": race.get("Circuit", {}).get("Location", {}).get("country", ""),
                "date": str(rd), "is_past": rd < today,
            })
        if rows:
            return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    except Exception as e:
        print(f"[f1] schedule error for {year}: {e}")
    return _static_schedule(year)

def _static_schedule(year):
    try:
        from utils.f1_data import ALL_RACES, FUTURE_RACES
        today = datetime.today().date()
        rows = []
        for r in ALL_RACES.get(year, []):
            rows.append({"round":r.get("round",0),"race":r.get("race",""),
                "circuit":r.get("circuit",""),"country":"","date":r.get("date",""),"is_past":True})
        if year == today.year:
            done = {r["round"] for r in rows}
            for r in (FUTURE_RACES or []):
                if r.get("round") in done:
                    continue
                rows.append({"round":r.get("round",0),"race":r.get("race",""),
                    "circuit":r.get("circuit",""),"country":r.get("country",""),
                    "date":r.get("date",""),"is_past":pd.to_datetime(r.get("date")).date() < today})
        return pd.DataFrame(rows).sort_values("round").reset_index(drop=True) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_data_source_status():
    status = {"jolpica": False, "fastf1": False, "yfinance": False}
    status["jolpica"] = _jolpica_json(f"{datetime.today().year}/races.json?limit=1", retries=1) is not None
    try:
        import fastf1  # noqa: F401
        status["fastf1"] = True
    except ImportError:
        pass
    try:
        status["yfinance"] = not yf.download("^GSPC", period="5d", progress=False).empty
    except Exception:
        pass
    return status

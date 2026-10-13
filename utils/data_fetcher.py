"""
data_fetcher.py — Dynamic 4-tier F1 + stock data fetcher

Tier 1: FastF1       — richest: lap times, sectors, pit stops, telemetry
Tier 2: OpenF1 API   — official modern REST API (free, no key)
Tier 3: Jolpica      — Ergast mirror with better uptime
Tier 4: Static cache — always works offline

Stock: yfinance (Yahoo Finance), synthetic fallback only if down.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import time
import os
import warnings
warnings.filterwarnings("ignore")

_DIR         = os.path.dirname(os.path.abspath(__file__))
FASTF1_CACHE = os.path.join(_DIR, "..", ".f1cache")
os.makedirs(FASTF1_CACHE, exist_ok=True)

OPENF1_BASE  = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
ERGAST_BASE  = "https://ergast.com/api/f1"
HEADERS      = {"User-Agent": "F1Pulse/1.0", "Accept": "application/json"}

# ── STOCK DATA ────────────────────────────────────────────────────────────────

def get_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [str(c) for c in df.columns]
        df = df.rename(columns={"index":"Date","Datetime":"Date"})
        if "Date" not in df.columns:
            df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        cols = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[cols].dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
        df["is_synthetic"] = False
        return df
    except Exception as e:
        print(f"[stock] yfinance error for {ticker}: {e}")
        return _synthetic_stock(ticker, start, end)

def _synthetic_stock(ticker, start, end):
    rng   = np.random.default_rng(abs(hash(ticker)) % (2**31))
    dates = pd.date_range(start=start, end=end, freq="B")
    n     = len(dates)
    if n == 0:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume","is_synthetic"])
    bases = {"RACE":370,"FWONA":82,"GOOGL":175,"GM":48,"ORCL":145,"AMZN":195,
             "HPQ":35,"V":285,"CRM":285,"TEAM":260,"AXP":235,"COIN":225,
             "FLUT":210,"PEP":155,"MGRM":4.5,"HMC":28,"F":11,"MBG":75}
    base  = bases.get(ticker.split(".")[0], 80.0)
    price = base * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    return pd.DataFrame({
        "Date":dates,"Open":price*rng.uniform(0.995,1.005,n),
        "High":price*rng.uniform(1.005,1.015,n),"Low":price*rng.uniform(0.985,0.995,n),
        "Close":price,"Volume":rng.integers(500_000,8_000_000,n).astype(float),
        "is_synthetic":True,
    })

# ── F1 EVENTS — 4-TIER ───────────────────────────────────────────────────────

def get_f1_events(start, end, team_filter=None):
    """Fetch F1 results dynamically. Each year tried independently."""
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    all_rows = []
    for year in range(max(start_dt.year, 2018), min(end_dt.year, datetime.today().year) + 1):
        rows = (_fetch_fastf1(year) or _fetch_openf1(year) or
                _fetch_ergast_base(year, JOLPICA_BASE, "jolpica") or
                _fetch_ergast_base(year, ERGAST_BASE,  "ergast")  or
                _fetch_static(year))
        all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    if team_filter:
        df = df[df["winner_team"].str.contains(team_filter, case=False, na=False)]
    return df.sort_values("date").reset_index(drop=True)

def _fetch_fastf1(year):
    try:
        import fastf1
        fastf1.Cache.enable_cache(FASTF1_CACHE)
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule is None or schedule.empty:
            return []
        rows  = []
        today = datetime.today().date()
        for _, event in schedule.iterrows():
            try:
                raw = event.get("EventDate") or event.get("Session5Date")
                if raw is None:
                    continue
                edate = pd.to_datetime(raw).date()
                if edate > today:
                    continue
                session = fastf1.get_session(year, int(event["RoundNumber"]), "R")
                session.load(telemetry=False, weather=False, messages=False, laps=True)
                res = session.results
                if res is None or res.empty:
                    continue
                res = res.sort_values("Position").reset_index(drop=True)
                p1  = res.iloc[0]
                p2r = res.iloc[1] if len(res) > 1 else None
                p3r = res.iloc[2] if len(res) > 2 else None
                fl  = ""
                try:
                    laps = session.laps
                    if laps is not None and not laps.empty:
                        fl = str(laps.loc[laps["LapTime"].idxmin(), "Driver"])
                except Exception:
                    pass
                pits = 0
                try:
                    wl   = session.laps.pick_driver(str(p1.get("Abbreviation","")))
                    pits = int(wl["PitInTime"].notna().sum())
                except Exception:
                    pass
                rows.append({
                    "date":str(edate), "race":str(event.get("EventName","")),
                    "circuit":str(event.get("Location","")),
                    "round":int(event.get("RoundNumber",0)), "season":year,
                    "winner":str(p1.get("FullName","")),
                    "winner_abbr":str(p1.get("Abbreviation","")),
                    "winner_team":str(p1.get("TeamName","")),
                    "winner_number":str(p1.get("DriverNumber","")),
                    "p2":str(p2r.get("FullName","")) if p2r is not None else "",
                    "p3":str(p3r.get("FullName","")) if p3r is not None else "",
                    "event_type":_cls_status(str(p1.get("Status","Finished")), 1),
                    "fastest_lap":fl, "winner_pits":pits,
                    "grid_position":int(p1.get("GridPosition",0) or 0),
                    "winner_points":float(p1.get("Points",25) or 25),
                    "data_source":"fastf1",
                })
            except Exception:
                continue
        return rows
    except Exception:
        return []

def _fetch_openf1(year):
    try:
        resp = requests.get(f"{OPENF1_BASE}/meetings?year={year}", headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        rows  = []
        today = datetime.today().date()
        for mtg in resp.json():
            try:
                ds = (mtg.get("date_start") or "")[:10]
                if not ds or pd.to_datetime(ds).date() > today:
                    continue
                mk  = mtg.get("meeting_key")
                sr  = requests.get(f"{OPENF1_BASE}/sessions?meeting_key={mk}&session_name=Race",
                                   headers=HEADERS, timeout=8)
                if sr.status_code != 200 or not sr.json():
                    continue
                sk = sr.json()[0].get("session_key")

                def _get_driver(pos_n):
                    pr = requests.get(f"{OPENF1_BASE}/position?session_key={sk}&position={pos_n}",
                                      headers=HEADERS, timeout=6)
                    if pr.status_code != 200 or not pr.json():
                        return {}
                    dn  = pr.json()[-1].get("driver_number")
                    dr  = requests.get(f"{OPENF1_BASE}/drivers?session_key={sk}&driver_number={dn}",
                                       headers=HEADERS, timeout=6)
                    return dr.json()[0] if dr.status_code==200 and dr.json() else {}

                p1i = _get_driver(1); p2i = _get_driver(2); p3i = _get_driver(3)
                rows.append({
                    "date":ds, "race":mtg.get("meeting_name",""),
                    "circuit":mtg.get("circuit_short_name",""),
                    "round":mk, "season":year,
                    "winner":p1i.get("full_name",""),
                    "winner_abbr":p1i.get("name_acronym",""),
                    "winner_team":p1i.get("team_name",""),
                    "winner_number":str(p1i.get("driver_number","")),
                    "p2":p2i.get("full_name",""), "p3":p3i.get("full_name",""),
                    "event_type":"WIN", "fastest_lap":"",
                    "winner_pits":0, "grid_position":0, "winner_points":25.0,
                    "data_source":"openf1",
                })
                time.sleep(0.15)
            except Exception:
                continue
        return rows
    except Exception:
        return []

def _fetch_ergast_base(year, base, src):
    try:
        resp = requests.get(f"{base}/{year}/results.json?limit=500", headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        races = resp.json()["MRData"]["RaceTable"]["Races"]
        rows  = []
        for race in races:
            results = race.get("Results", [])
            if not results:
                continue
            def dn(r):
                d = r.get("Driver",{})
                return f"{d.get('givenName','')} {d.get('familyName','')}".strip()
            p1,p2,p3 = (results[i] if len(results)>i else {} for i in range(3))
            fl = (p1.get("FastestLap") or {})
            rows.append({
                "date":race.get("date",""), "race":race.get("raceName",""),
                "circuit":race.get("Circuit",{}).get("circuitName",""),
                "round":int(race.get("round",0)), "season":year,
                "winner":dn(p1),
                "winner_abbr":(p1.get("Driver") or {}).get("code",""),
                "winner_team":(p1.get("Constructor") or {}).get("name",""),
                "winner_number":p1.get("number",""),
                "p2":dn(p2), "p3":dn(p3),
                "event_type":_cls_ergast(p1),
                "fastest_lap":dn(p1) if fl.get("rank")=="1" else "",
                "winner_pits":0,
                "grid_position":int(p1.get("grid",0) or 0),
                "winner_points":float(p1.get("points",0) or 0),
                "data_source":src,
            })
        return rows
    except Exception:
        return []

def _fetch_static(year):
    try:
        from utils.f1_data import ALL_RACES
        rows = []
        for r in ALL_RACES.get(year, []):
            row = dict(r)
            for k in ["winner_abbr","winner_number"]:
                row.setdefault(k,"")
            for k in ["winner_pits","grid_position"]:
                row.setdefault(k, 0)
            row.setdefault("winner_points", 25.0)
            row.setdefault("fastest_lap", "")
            row["data_source"] = "static_cache"
            rows.append(row)
        return rows
    except Exception:
        return []

def _cls_status(status, pos):
    s = status.lower()
    if any(w in s for w in ["accident","collision","crash"]):
        return "CRASH"
    if pos == 1:   return "WIN"
    if pos <= 3:   return "PODIUM"
    if any(w in s for w in ["retired","dnf","mechanical"]): return "DNF"
    return "RACE"

def _cls_ergast(result):
    s   = str(result.get("status","")).lower()
    pos = int(result.get("position",99) or 99)
    if any(w in s for w in ["accident","collision"]): return "CRASH"
    if pos == 1:   return "WIN"
    if pos <= 3:   return "PODIUM"
    if "retired" in s or pos > 20: return "DNF"
    return "RACE"

# ── LIVE STANDINGS & SCHEDULE ─────────────────────────────────────────────────

def get_driver_standings(year):
    for base in [JOLPICA_BASE, ERGAST_BASE]:
        try:
            resp = requests.get(f"{base}/{year}/driverStandings.json", headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
            lists = resp.json()["MRData"]["StandingsTable"]["StandingsLists"]
            if not lists:
                continue
            rows = []
            for s in lists[0]["DriverStandings"]:
                d = s.get("Driver",{})
                c = (s.get("Constructors") or [{}])[0]
                rows.append({
                    "position":int(s.get("position",99)),
                    "driver":f"{d.get('givenName','')} {d.get('familyName','')}".strip(),
                    "team":c.get("name",""),
                    "points":float(s.get("points",0)),
                    "wins":int(s.get("wins",0)),
                    "nationality":d.get("nationality",""),
                })
            if rows:
                return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
        except Exception:
            continue
    return _static_standings(year)

def _static_standings(year):
    try:
        from utils.f1_data import DRIVER_STANDINGS
        df = pd.DataFrame(DRIVER_STANDINGS.get(year, []))
        if not df.empty:
            df["position"] = range(1, len(df)+1)
            df["nationality"] = ""
        return df
    except Exception:
        return pd.DataFrame()

def get_constructor_standings(year):
    for base in [JOLPICA_BASE, ERGAST_BASE]:
        try:
            resp = requests.get(f"{base}/{year}/constructorStandings.json", headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
            lists = resp.json()["MRData"]["StandingsTable"]["StandingsLists"]
            if not lists:
                continue
            rows = []
            for s in lists[0]["ConstructorStandings"]:
                c = s.get("Constructor",{})
                rows.append({
                    "position":int(s.get("position",99)),
                    "team":c.get("name",""),
                    "points":float(s.get("points",0)),
                    "wins":int(s.get("wins",0)),
                })
            if rows:
                return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame()

def get_race_schedule(year):
    for base in [JOLPICA_BASE, ERGAST_BASE]:
        try:
            resp = requests.get(f"{base}/{year}/races.json?limit=30", headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
            races = resp.json()["MRData"]["RaceTable"]["Races"]
            today = datetime.today().date()
            rows  = []
            for race in races:
                rd = pd.to_datetime(race.get("date","")).date()
                rows.append({
                    "round":int(race.get("round",0)),
                    "race":race.get("raceName",""),
                    "circuit":race.get("Circuit",{}).get("circuitName",""),
                    "country":race.get("Circuit",{}).get("Location",{}).get("country",""),
                    "date":str(rd), "is_past": rd <= today,
                })
            if rows:
                return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
        except Exception:
            continue
    return _static_schedule(year)

def _static_schedule(year):
    try:
        from utils.f1_data import ALL_RACES, FUTURE_RACES
        rows = []
        for r in ALL_RACES.get(year, []):
            rows.append({"round":r.get("round",0),"race":r.get("race",""),
                "circuit":r.get("circuit",""),"country":"","date":r.get("date",""),"is_past":True})
        if year == datetime.today().year:
            for r in (FUTURE_RACES or []):
                rows.append({"round":r.get("round",0),"race":r.get("race",""),
                    "circuit":r.get("circuit",""),"country":r.get("country",""),
                    "date":r.get("date",""),"is_past":False})
        return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def get_data_source_status():
    status = {"fastf1":False,"openf1":False,"jolpica":False,"ergast":False,"yfinance":False}
    try:
        import fastf1; status["fastf1"] = True
    except ImportError:
        pass
    for key, url in [
        ("openf1",  f"{OPENF1_BASE}/sessions?year=2024&session_name=Race"),
        ("jolpica", f"{JOLPICA_BASE}/2024/races.json?limit=1"),
        ("ergast",  f"{ERGAST_BASE}/2024/races.json?limit=1"),
    ]:
        try:
            status[key] = requests.get(url, headers=HEADERS, timeout=5).status_code == 200
        except Exception:
            pass
    try:
        status["yfinance"] = not yf.download("AAPL", period="1d", progress=False).empty
    except Exception:
        pass
    return status

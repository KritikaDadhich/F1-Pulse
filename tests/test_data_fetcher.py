import json
import os
from datetime import datetime

import pytest

from utils import data_fetcher as df
from utils.teams_sponsors import TEAM_COLORS


def result(pos, team, status="Finished", classified=True, points=0.0, driver=None):
    return {"position": pos, "classified": classified, "status": status, "driver": driver or f"Driver {pos}",
            "abbr": "", "number": "", "team": team, "grid": pos, "points": points, "fastest": False}


def test_race_row_summarises_each_team():
    results = [result(2, "McLaren", points=18), result(1, "Ferrari", points=25),
               result(3, "McLaren", points=15), result(20, "Haas", "Retired", classified=False),
               result(12, "Haas", "Lapped")]
    row = df._race_row(results, 2026, 5, "2026-05-24", "Canadian Grand Prix", "Montreal", "Canada", "jolpica")
    assert row["winner_team"] == "Ferrari"
    assert (row["p2_team"], row["p3_team"]) == ("McLaren", "McLaren")
    assert row["team_results"]["McLaren"] == {"best": 2, "dnf": False, "points": 33.0,
                                              "drivers": ["Driver 2", "Driver 3"]}
    assert row["team_results"]["Haas"]["dnf"] is True
    assert row["team_results"]["Haas"]["best"] == 12


@pytest.mark.parametrize("status,classified,expected", [
    ("Finished", True, False), ("+1 Lap", True, False), ("Lapped", True, False),
    ("Retired", False, True), ("Retired", True, True), ("Disqualified", False, True),
    ("Did not start", False, True), ("Collision", False, True),
])
def test_dnf_detection(status, classified, expected):
    assert df._is_dnf(result(15, "Williams", status, classified)) is expected


def test_static_fallback_has_full_schema():
    rows = df._fetch_static(2024)
    assert len(rows) == 24
    for key in ("date", "winner_team", "team_results", "p2_team", "data_source"):
        assert key in rows[0]


@pytest.mark.parametrize("year", range(df.FIRST_SEASON, datetime.today().year))
def test_bundled_finished_seasons(year):
    """data/f1_seasons/ must hold every finished season, in the current row format."""
    path = df._season_file(df.BUNDLED_SEASONS, year)
    assert os.path.exists(path), f"missing {path} — run python train_model.py"
    rows = json.load(open(path, encoding="utf-8"))
    assert len(rows) >= 20
    assert [r["round"] for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert r["season"] == year and r["data_source"] in ("jolpica", "fastf1")
        assert r["winner_team"] in r["team_results"]
        for team, res in r["team_results"].items():
            assert team in TEAM_COLORS, f"{year} R{r['round']}: unknown team name {team!r}"
            assert {"best", "dnf", "points", "drivers"} <= set(res)


def test_finished_season_served_without_network(monkeypatch):
    monkeypatch.setattr(df, "_fetch_jolpica", lambda y: pytest.fail("should not call the API"))
    rows = df.get_season_results(df.FIRST_SEASON)
    assert len(rows) >= 20


def test_failed_stock_download_returns_empty_frame(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(df.yf, "download", boom)
    out = df.get_stock_data("RACE", "2026-01-01", "2026-02-01")
    assert out.empty and "Close" in out.columns

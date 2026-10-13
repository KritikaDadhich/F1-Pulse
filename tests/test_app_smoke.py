"""
Runs the whole Streamlit app headlessly with network calls replaced by local data:
finished seasons come from data/f1_seasons/, the current season from the bundled
fallback, and prices from a deterministic test series.
"""
import os
import socket

import pytest
from streamlit.testing.v1 import AppTest

from conftest import make_prices, ROOT
import utils.data_fetcher as df


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(df, "get_stock_data", lambda t, s, e: make_prices(start=s, seed=len(t)))
    monkeypatch.setattr(df, "get_stock_currency", lambda t, fallback="USD": fallback)
    monkeypatch.setattr(df, "_jolpica_json", lambda path, retries=5: None)
    monkeypatch.setattr(df, "_fetch_fastf1", lambda year: [])
    monkeypatch.setattr(df, "get_data_source_status",
                        lambda: {"jolpica": False, "fastf1": True, "yfinance": False})
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    def no_network(*a, **k):
        raise AssertionError("test tried to use the network")
    monkeypatch.setattr(socket.socket, "connect", no_network)


def run(team=None, company_index=0):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
    if team:
        at.sidebar.selectbox[0].set_value(team).run()
        options = at.sidebar.selectbox[1].options
        at.sidebar.selectbox[1].set_value(options[company_index % len(options)]).run()
    return at


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("team", ["Ferrari", "Audi", "Cadillac", "Racing Bulls"])
def test_app_renders_for_team(team):
    at = run(team)
    assert_clean(at)
    assert len(at.tabs) == 7


def test_prediction_button():
    at = run("Mercedes")
    at.button[0].click().run()
    assert_clean(at)
    assert [m.label for m in at.metric][0] == "Predicted direction"


def test_season_selectors():
    at = run()
    for key_or_label in ("stand_yr", "Select season"):
        box = (at.selectbox(key=key_or_label) if key_or_label == "stand_yr"
               else next(s for s in at.selectbox if s.label == key_or_label))
        for year in box.options:
            (at.selectbox(key=key_or_label) if key_or_label == "stand_yr"
             else next(s for s in at.selectbox if s.label == key_or_label)).set_value(year).run()
            assert_clean(at)

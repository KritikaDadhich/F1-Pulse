from utils.teams_sponsors import (
    F1_TEAMS_2026, EXCHANGE_INFO, TEAM_COLORS, GLOBAL_F1_SPONSORS,
    canonical_team, team_matches, current_team_for, get_yfinance_ticker, get_all_public_companies,
)


def test_api_names_map_to_app_names():
    assert canonical_team("Red Bull") == "Red Bull Racing"
    assert canonical_team("RB F1 Team") == "Racing Bulls"
    assert canonical_team("Haas F1 Team") == "Haas"
    assert canonical_team("Alpine F1 Team") == "Alpine"
    assert canonical_team("Cadillac F1 Team") == "Cadillac"
    assert canonical_team("Ferrari") == "Ferrari"


def test_team_lineage():
    assert team_matches("Racing Bulls", "AlphaTauri")
    assert team_matches("Audi", "Sauber") and team_matches("Audi", "Alfa Romeo")
    assert not team_matches("Ferrari", "Haas")
    assert current_team_for("AlphaTauri") == "Racing Bulls"


def test_yfinance_suffixes():
    assert get_yfinance_ticker("RACE", "NYSE") == "RACE"
    assert get_yfinance_ticker("MBG.DE", "XETRA") == "MBG.DE"
    assert get_yfinance_ticker("SHEL", "LSE") == "SHEL.L"
    assert get_yfinance_ticker("HEIA", "AMS") == "HEIA.AS"
    assert get_yfinance_ticker("7203.T", "TSE") == "7203.T"


def test_every_company_has_a_known_exchange():
    for c in get_all_public_companies():
        assert c["exchange"] in EXCHANGE_INFO, c
        assert EXCHANGE_INFO[c["exchange"]].get("index"), c["exchange"]


def test_no_delisted_tickers():
    tickers = {c["ticker"] for c in get_all_public_companies()}
    assert not tickers & {"MGRM", "DPW.DE", "992.HK"}


def test_every_team_is_complete():
    for name, team in F1_TEAMS_2026.items():
        assert name in TEAM_COLORS
        assert len(team["drivers_2026"]) == 2
        for sp in team["key_public_sponsors"]:
            assert bool(sp.get("ticker")) == bool(sp.get("exchange")), (name, sp["company"])


def test_global_sponsor_exchanges():
    for sp in GLOBAL_F1_SPONSORS:
        if sp.get("public"):
            assert sp["exchange"] in EXCHANGE_INFO, sp["company"]

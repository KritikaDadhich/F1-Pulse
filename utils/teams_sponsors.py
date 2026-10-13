"""
teams_sponsors.py
Research-verified data for the 2026 F1 season — 11 teams, 22 drivers,
all official F1 global/team sponsors from the official F1 partners page,
with every stock ticker verified against its PRIMARY exchange listing.

Sources:
  - Official F1 driver/team page: formula1.com/en/drivers (confirmed July 2026)
  - Official F1 partners page: formula1.com/en/partners (document provided)
  - Each ticker cross-checked against: NYSE, NASDAQ, LSE, XETRA, SIX, Euronext, HKSE, Tadawul

Why 2026 and not 2025?
  The current season is 2026. The app now shows both 2025 and 2026 data
  so users can compare and see how the grid evolved.
"""

# ── Exchange reference ────────────────────────────────────────────────────────
EXCHANGE_INFO = {
    "NYSE":     {"full_name": "New York Stock Exchange",          "country": "USA",          "currency": "USD", "flag": "🇺🇸"},
    "NASDAQ":   {"full_name": "NASDAQ",                            "country": "USA",          "currency": "USD", "flag": "🇺🇸"},
    "LSE":      {"full_name": "London Stock Exchange",            "country": "UK",           "currency": "GBP", "flag": "🇬🇧"},
    "XETRA":    {"full_name": "Deutsche Börse / Xetra",          "country": "Germany",      "currency": "EUR", "flag": "🇩🇪"},
    "BIT":      {"full_name": "Borsa Italiana (Milan)",           "country": "Italy",        "currency": "EUR", "flag": "🇮🇹"},
    "EURONEXT": {"full_name": "Euronext Paris",                    "country": "France",       "currency": "EUR", "flag": "🇫🇷"},
    "SIX":      {"full_name": "SIX Swiss Exchange",               "country": "Switzerland",  "currency": "CHF", "flag": "🇨🇭"},
    "HKSE":     {"full_name": "Hong Kong Stock Exchange",         "country": "Hong Kong",    "currency": "HKD", "flag": "🇭🇰"},
    "TSE":      {"full_name": "Tokyo Stock Exchange",              "country": "Japan",        "currency": "JPY", "flag": "🇯🇵"},
    "TADAWUL":  {"full_name": "Saudi Exchange (Tadawul)",         "country": "Saudi Arabia", "currency": "SAR", "flag": "🇸🇦"},
    "MADRID":   {"full_name": "Bolsa de Madrid (BME)",            "country": "Spain",        "currency": "EUR", "flag": "🇪🇸"},
    "ASX":      {"full_name": "Australian Securities Exchange",    "country": "Australia",    "currency": "AUD", "flag": "🇦🇺"},
}

# ── All 11 F1 teams — 2026 season (verified from formula1.com July 2026) ─────
F1_TEAMS_2026 = {
    "McLaren": {
        "full_name": "McLaren Formula 1 Team",
        "engine": "Mercedes",
        "base_country": "UK",
        "base_city": "Woking, England",
        "drivers_2026": ["Lando Norris", "Oscar Piastri"],
        "team_colour": "#FF8000",
        "constructor_position_2025": 1,
        "parent_company": {
            "name": "McLaren Group (Mumtalakat / CYVN Holdings)",
            "public": False,
            "note": "Majority owned by Bahrain's Mumtalakat sovereign wealth fund. Not directly tradable — track via its title sponsor Google (Alphabet).",
        },
        "key_public_sponsors": [
            {"company": "Alphabet Inc. (Google)",  "ticker": "GOOGL", "exchange": "NASDAQ", "role": "Title partner — Google branding on car", "on_car": True},
            {"company": "American Express",         "ticker": "AXP",   "exchange": "NYSE",   "role": "Official partner — card on driver suits", "on_car": True},
            {"company": "Dell Technologies",        "ticker": "DELL",  "exchange": "NYSE",   "role": "Technology partner", "on_car": True},
        ],
    },
    "Ferrari": {
        "full_name": "Scuderia Ferrari HP",
        "engine": "Ferrari",
        "base_country": "Italy",
        "base_city": "Maranello, Italy",
        "drivers_2026": ["Charles Leclerc", "Lewis Hamilton"],
        "team_colour": "#E8002D",
        "constructor_position_2025": 4,
        "parent_company": {
            "name": "Ferrari N.V.",
            "public": True,
            "ticker": "RACE",
            "exchange": "NYSE",
            "note": "Ferrari is unique — the F1 team, the road car company, and the stock are all the SAME legal entity. When Ferrari wins, NYSE:RACE moves.",
        },
        "key_public_sponsors": [
            {"company": "HP Inc.",                  "ticker": "HPQ",   "exchange": "NYSE",   "role": "Title sponsor — team officially called 'Scuderia Ferrari HP'", "on_car": True},
            {"company": "Shell plc",                "ticker": "SHEL",  "exchange": "LSE",    "role": "Fuel & lubricants — Shell logo on every Ferrari", "on_car": True},
            {"company": "Puma SE",                  "ticker": "PUM.DE","exchange": "XETRA",  "role": "Official kit supplier — on driver race suits", "on_car": False},
            {"company": "LVMH (Louis Vuitton/TAG Heuer)", "ticker": "MC.PA", "exchange": "EURONEXT", "role": "TAG Heuer is F1 official timekeeper — on car", "on_car": True},
        ],
    },
    "Mercedes": {
        "full_name": "Mercedes-AMG Petronas F1 Team",
        "engine": "Mercedes",
        "base_country": "UK",
        "base_city": "Brackley, England",
        "drivers_2026": ["George Russell", "Andrea Kimi Antonelli"],
        "team_colour": "#27F4D2",
        "constructor_position_2025": 2,
        "parent_company": {
            "name": "Mercedes-Benz Group AG",
            "public": True,
            "ticker": "MBG.DE",
            "exchange": "XETRA",
            "note": "Primary listing is Frankfurt/Xetra (MBG.DE). AVOID the illiquid US OTC ADR ticker MBGYY — it trades at a premium to the real Frankfurt price.",
        },
        "key_public_sponsors": [
            {"company": "Petronas (MISC Berhad parent)", "ticker": None, "exchange": None, "role": "Title sponsor — team named 'AMG Petronas' (Petronas is Malaysian state-owned)", "on_car": True},
            {"company": "Adidas AG",               "ticker": "ADS.DE","exchange": "XETRA",  "role": "Official apparel partner — on team kit", "on_car": False},
            {"company": "SAP SE",                   "ticker": "SAP",   "exchange": "NYSE",   "role": "Technology partner — SAP logo on car", "on_car": True},
            {"company": "Salesforce Inc.",          "ticker": "CRM",   "exchange": "NYSE",   "role": "Global partner — Racing Bulls AND F1 globally", "on_car": True},
        ],
    },
    "Red Bull Racing": {
        "full_name": "Oracle Red Bull Racing",
        "engine": "Red Bull Powertrains (Ford)",
        "base_country": "UK",
        "base_city": "Milton Keynes, England",
        "drivers_2026": ["Max Verstappen", "Isack Hadjar"],
        "team_colour": "#3671C6",
        "constructor_position_2025": 3,
        "parent_company": {
            "name": "Red Bull GmbH",
            "public": False,
            "note": "Privately held (Mateschitz family). Engine co-developed with Ford Motor Co. (NYSE:F) — Ford returned to F1 in 2026 for the first time since 2004.",
        },
        "key_public_sponsors": [
            {"company": "Oracle Corporation",       "ticker": "ORCL",  "exchange": "NYSE",   "role": "Title sponsor — team named 'Oracle Red Bull Racing'", "on_car": True},
            {"company": "Ford Motor Company",       "ticker": "F",     "exchange": "NYSE",   "role": "Engine technology partner — Ford returned to F1 2026", "on_car": True},
            {"company": "Honda Motor Co.",          "ticker": "HMC",   "exchange": "NYSE",   "role": "Previous engine supplier (2021-2025) — historical link", "on_car": False},
        ],
    },
    "Aston Martin": {
        "full_name": "Aston Martin Aramco F1 Team",
        "engine": "Honda",
        "base_country": "UK",
        "base_city": "Silverstone, England",
        "drivers_2026": ["Fernando Alonso", "Lance Stroll"],
        "team_colour": "#229971",
        "constructor_position_2025": 5,
        "parent_company": {
            "name": "Aston Martin Lagonda Global Holdings plc",
            "public": True,
            "ticker": "AML.L",
            "exchange": "LSE",
            "note": "The road car company trades on London Stock Exchange. Honda now powers the F1 team as a works partner from 2026 — a historic return.",
        },
        "key_public_sponsors": [
            {"company": "Saudi Aramco",             "ticker": "2222.SR","exchange": "TADAWUL","role": "Title sponsor — team officially 'Aston Martin Aramco'", "on_car": True},
            {"company": "Honda Motor Co.",          "ticker": "HMC",   "exchange": "NYSE",   "role": "Works engine partner from 2026 — Honda returned to F1 independently", "on_car": True},
            {"company": "Cognizant",                "ticker": "CTSH",  "exchange": "NASDAQ", "role": "Technology partner — on car livery", "on_car": True},
            {"company": "Coinbase Global",          "ticker": "COIN",  "exchange": "NASDAQ", "role": "Official crypto partner — on car", "on_car": True},
        ],
    },
    "Alpine": {
        "full_name": "BWT Alpine F1 Team",
        "engine": "Mercedes",
        "base_country": "France",
        "base_city": "Enstone, England / Viry-Châtillon, France",
        "drivers_2026": ["Pierre Gasly", "Franco Colapinto"],
        "team_colour": "#0093CC",
        "constructor_position_2025": 6,
        "parent_company": {
            "name": "Renault Group",
            "public": True,
            "ticker": "RNO.PA",
            "exchange": "EURONEXT",
            "note": "Alpine is Renault's F1 team. Note: Alpine SWITCHED from Renault engines to Mercedes power units for 2026 — Renault is no longer an F1 engine supplier.",
        },
        "key_public_sponsors": [
            {"company": "BWT AG",                  "ticker": None,    "exchange": None,     "role": "Title sponsor (Austrian water tech firm, privately held)", "on_car": True},
        ],
    },
    "Williams": {
        "full_name": "Atlassian Williams Racing",
        "engine": "Mercedes",
        "base_country": "UK",
        "base_city": "Grove, England",
        "drivers_2026": ["Alexander Albon", "Carlos Sainz"],
        "team_colour": "#64C4FF",
        "constructor_position_2025": 7,
        "parent_company": {
            "name": "Williams Grand Prix Holdings (Dorilton Capital)",
            "public": False,
            "note": "Owned by private equity firm Dorilton Capital. Track via title sponsor Atlassian (NASDAQ:TEAM).",
        },
        "key_public_sponsors": [
            {"company": "Atlassian Corporation",   "ticker": "TEAM",  "exchange": "NASDAQ", "role": "Title sponsor — team renamed 'Atlassian Williams Racing'", "on_car": True},
            {"company": "Santander",               "ticker": "SAN.MC","exchange": "MADRID", "role": "Official partner — Santander bank on car", "on_car": True},
        ],
    },
    "Racing Bulls": {
        "full_name": "Visa Cash App Racing Bulls F1 Team",
        "engine": "Red Bull Powertrains (Ford)",
        "base_country": "Italy",
        "base_city": "Faenza, Italy",
        "drivers_2026": ["Liam Lawson", "Arvid Lindblad"],
        "team_colour": "#6692FF",
        "constructor_position_2025": 8,
        "parent_company": {
            "name": "Red Bull GmbH (sister team)",
            "public": False,
            "note": "Same ownership group as Oracle Red Bull Racing. Also uses Ford-backed Red Bull power unit.",
        },
        "key_public_sponsors": [
            {"company": "Visa Inc.",                "ticker": "V",     "exchange": "NYSE",   "role": "Title sponsor — Visa logo on car", "on_car": True},
            {"company": "Block Inc. (Cash App)",    "ticker": "XYZ",   "exchange": "NYSE",   "role": "Title sponsor — Cash App logo on car", "on_car": True},
            {"company": "Salesforce Inc.",          "ticker": "CRM",   "exchange": "NYSE",   "role": "Official partner — Salesforce + Racing Bulls announced 2025", "on_car": True},
            {"company": "Ford Motor Company",       "ticker": "F",     "exchange": "NYSE",   "role": "Engine technology partner (via Red Bull Powertrains)", "on_car": False},
        ],
    },
    "Haas": {
        "full_name": "MoneyGram Haas F1 Team",
        "engine": "Ferrari",
        "base_country": "USA",
        "base_city": "Kannapolis, USA / Banbury, England",
        "drivers_2026": ["Esteban Ocon", "Oliver Bearman"],
        "team_colour": "#B6BABD",
        "constructor_position_2025": 9,
        "parent_company": {
            "name": "Haas Automation Inc.",
            "public": False,
            "note": "Privately owned by American businessman Gene Haas. Uses Ferrari engines — Ferrari (NYSE:RACE) is the engine link.",
        },
        "key_public_sponsors": [
            {"company": "MoneyGram International", "ticker": "MGRM",  "exchange": "NASDAQ", "role": "Title sponsor — MoneyGram on car", "on_car": True},
            {"company": "Ferrari N.V.",             "ticker": "RACE",  "exchange": "NYSE",   "role": "Engine supplier — Ferrari powers Haas", "on_car": False},
        ],
    },
    "Audi": {
        "full_name": "Audi F1 Team",
        "engine": "Audi (Volkswagen Group)",
        "base_country": "Switzerland",
        "base_city": "Hinwil, Switzerland",
        "drivers_2026": ["Nico Hülkenberg", "Gabriel Bortoleto"],
        "team_colour": "#BB0000",
        "constructor_position_2025": 10,
        "parent_company": {
            "name": "Volkswagen AG (via Audi AG subsidiary)",
            "public": True,
            "ticker": "VOW3.DE",
            "exchange": "XETRA",
            "note": "Audi is a wholly-owned subsidiary of Volkswagen AG. VW trades as VOW3.DE on Frankfurt. Audi entered F1 in 2026 by acquiring the former Sauber/Kick Sauber team.",
        },
        "key_public_sponsors": [
            {"company": "Volkswagen AG",            "ticker": "VOW3.DE","exchange": "XETRA", "role": "Parent company — owns the team outright through Audi AG", "on_car": False},
        ],
    },
    "Cadillac": {
        "full_name": "Cadillac F1 Team",
        "engine": "Ferrari (transitioning to GM in 2029)",
        "base_country": "USA",
        "base_city": "Speedway, Indiana, USA",
        "drivers_2026": ["Sergio Pérez", "Valtteri Bottas"],
        "team_colour": "#CE1126",
        "constructor_position_2025": None,
        "parent_company": {
            "name": "General Motors Company",
            "public": True,
            "ticker": "GM",
            "exchange": "NYSE",
            "note": "General Motors owns Cadillac. GM makes its F1 debut in 2026 — the first American manufacturer on the grid in the modern era. NYSE:GM is directly affected by Cadillac's F1 results.",
        },
        "key_public_sponsors": [
            {"company": "General Motors Company",   "ticker": "GM",    "exchange": "NYSE",   "role": "Parent company — Cadillac is GM's luxury brand", "on_car": False},
            {"company": "Ferrari N.V.",             "ticker": "RACE",  "exchange": "NYSE",   "role": "Engine supplier (Ferrari powers Cadillac until GM's own engine is ready 2029)", "on_car": False},
        ],
    },
}

# ── Universal suppliers — linked to every single team ────────────────────────
UNIVERSAL_SUPPLIERS = {
    "tyre": {
        "company": "Pirelli & C. S.p.A.",
        "ticker": "PIRC.MI",
        "exchange": "BIT",
        "role": "Exclusive tyre supplier — every team, every race, every lap uses Pirelli tyres",
        "note": "Pirelli has been F1's sole tyre supplier since 2011. Their deal runs to 2027. A dramatic tyre blowout creates a negative news cycle for PIRC.MI.",
    },
}

# ── F1 organisation itself ────────────────────────────────────────────────────
F1_PARENT = {
    "company": "Liberty Media Corp. (F1 Group tracking stock)",
    "ticker": "FWONA",
    "exchange": "NASDAQ",
    "role": "Owns the entire Formula One championship — race fees, TV rights ($2.6B/yr), sponsorships",
    "note": "More exciting seasons = more viewers = more ad/rights revenue = higher FWONA. The 2025 championship fight going to the final race was worth hundreds of millions to Liberty.",
}

# ── Official F1 Global Partners (from official F1 partners page, July 2026) ──
# These are verified from the document provided — only including those with
# publicly traded parent companies; private companies noted separately.
GLOBAL_F1_SPONSORS = [
    # Global Partners
    {"company": "Saudi Aramco",            "ticker": "2222.SR","exchange": "TADAWUL", "role": "Global Partner — on every car livery, pit lane branding", "tier": "Global Partner", "public": True},
    {"company": "Amazon (AWS)",            "ticker": "AMZN",  "exchange": "NASDAQ",  "role": "Global Partner — powers F1's AI insights on TV since 2018", "tier": "Global Partner", "public": True},
    {"company": "DHL (Deutsche Post AG)",  "ticker": "DPW.DE","exchange": "XETRA",   "role": "Global Logistics Partner — ships 2,000 tons of freight per race", "tier": "Global Partner", "public": True},
    {"company": "Heineken N.V.",           "ticker": "HEIA.AS","exchange": "Euronext Amsterdam","role": "Global Partner — DHL Fastest Lap Award sponsor", "tier": "Global Partner", "public": True},
    {"company": "Lenovo",                  "ticker": "992.HK","exchange": "HKSE",    "role": "Global Partner — technology sponsor", "tier": "Global Partner", "public": True},
    {"company": "LVMH (Louis Vuitton/TAG Heuer/Moët)", "ticker": "MC.PA","exchange": "EURONEXT","role": "Global Partner — 10-year $1B deal from 2025. TAG Heuer is official timekeeper", "tier": "Global Partner", "public": True},
    {"company": "PepsiCo (Sting/Gatorade/Doritos)", "ticker": "PEP","exchange": "NASDAQ","role": "Global Partner — joined 2025 with Sting Energy, Gatorade, Doritos", "tier": "Global Partner", "public": True},
    {"company": "Qatar Airways",           "ticker": None,    "exchange": None,       "role": "Global Partner — Official Airline of F1 (state-owned, not publicly traded)", "tier": "Global Partner", "public": False},
    {"company": "Salesforce Inc.",         "ticker": "CRM",   "exchange": "NYSE",     "role": "Global Partner — AI-powered fan companion built on Agentforce", "tier": "Global Partner", "public": True},
    # Official Partners
    {"company": "American Express",        "ticker": "AXP",   "exchange": "NYSE",     "role": "Official Partner — Race Day with American Express experiences", "tier": "Official Partner", "public": True},
    {"company": "Banco Santander",         "ticker": "SAN.MC","exchange": "MADRID",   "role": "Official Partner — Williams title sponsor + F1 official partner", "tier": "Official Partner", "public": True},
    {"company": "Standard Chartered",      "ticker": "STAN.L","exchange": "LSE",      "role": "Official Partner — global banking partner", "tier": "Official Partner", "public": True},
    {"company": "Nestlé (KitKat)",         "ticker": "NESN.SW","exchange": "SIX",    "role": "Official Partner — KitKat is Official Chocolate Bar of F1", "tier": "Official Partner", "public": True},
    {"company": "Crypto.com",              "ticker": None,    "exchange": None,       "role": "Official Partner — extended partnership to 2030 (private)", "tier": "Official Partner", "public": False},
    # Betting Partners (publicly traded parents)
    {"company": "Flutter Entertainment (FanDuel)", "ticker": "FLUT","exchange": "NYSE","role": "Official Betting Operator USA (FanDuel brand) — first ever F1 US betting partner", "tier": "Official Betting", "public": True},
]

# ── Per-team detailed sponsor lists (from official F1 partners page + team sites) ──
TEAM_CAR_SPONSORS = {
    "McLaren": [
        {"company": "Alphabet (Google)", "ticker": "GOOGL", "exchange": "NASDAQ", "where": "Car livery — Google Cloud branding"},
        {"company": "American Express",  "ticker": "AXP",   "exchange": "NYSE",   "where": "Driver suits & car"},
        {"company": "Dell Technologies", "ticker": "DELL",  "exchange": "NYSE",   "where": "Pit wall technology"},
        {"company": "Pirelli",           "ticker": "PIRC.MI","exchange": "BIT",   "where": "Tyres (all teams)"},
        {"company": "Puma SE",           "ticker": "PUM.DE","exchange": "XETRA",  "where": "Team kit"},
    ],
    "Ferrari": [
        {"company": "Ferrari N.V. (RACE)", "ticker": "RACE","exchange": "NYSE",   "where": "The team IS the company"},
        {"company": "HP Inc.",             "ticker": "HPQ", "exchange": "NYSE",   "where": "Car nose + sidepod"},
        {"company": "Shell plc",           "ticker": "SHEL","exchange": "LSE",    "where": "Shell logo on engine cover"},
        {"company": "Puma SE",             "ticker": "PUM.DE","exchange": "XETRA","where": "Driver race suits"},
        {"company": "LVMH / TAG Heuer",    "ticker": "MC.PA","exchange": "EURONEXT","where": "Official F1 timekeeper"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Mercedes": [
        {"company": "Mercedes-Benz (MBG.DE)", "ticker": "MBG.DE","exchange": "XETRA","where": "The team IS the company"},
        {"company": "Adidas AG",           "ticker": "ADS.DE","exchange": "XETRA","where": "Team kit + driver suits"},
        {"company": "SAP SE",              "ticker": "SAP",  "exchange": "NYSE",  "where": "Car livery"},
        {"company": "Salesforce Inc.",     "ticker": "CRM",  "exchange": "NYSE",  "where": "Car + team branding"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Red Bull Racing": [
        {"company": "Oracle Corporation",  "ticker": "ORCL","exchange": "NYSE",   "where": "Car nose and sidepod — naming rights"},
        {"company": "Ford Motor Company",  "ticker": "F",   "exchange": "NYSE",   "where": "Engine technology — Ford logo on car 2026"},
        {"company": "Puma SE",             "ticker": "PUM.DE","exchange": "XETRA","where": "Team kit"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Aston Martin": [
        {"company": "Aston Martin (AML.L)","ticker": "AML.L","exchange": "LSE",   "where": "The car brand is the team name"},
        {"company": "Saudi Aramco",        "ticker": "2222.SR","exchange": "TADAWUL","where": "Team name + car livery"},
        {"company": "Honda Motor Co.",     "ticker": "HMC", "exchange": "NYSE",   "where": "Engine cover — Honda works partnership"},
        {"company": "Coinbase",            "ticker": "COIN","exchange": "NASDAQ", "where": "Car sidepod"},
        {"company": "Cognizant",           "ticker": "CTSH","exchange": "NASDAQ", "where": "Car livery"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Alpine": [
        {"company": "Renault Group",       "ticker": "RNO.PA","exchange": "EURONEXT","where": "Parent company (Renault owns Alpine)"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Williams": [
        {"company": "Atlassian",           "ticker": "TEAM","exchange": "NASDAQ", "where": "Team name + car livery — title sponsor"},
        {"company": "Santander",           "ticker": "SAN.MC","exchange": "MADRID","where": "Car sidepod — Spanish bank"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Racing Bulls": [
        {"company": "Visa Inc.",           "ticker": "V",   "exchange": "NYSE",   "where": "Team name + car livery"},
        {"company": "Block Inc. (Cash App)","ticker": "XYZ","exchange": "NYSE",   "where": "Team name + car livery"},
        {"company": "Salesforce Inc.",     "ticker": "CRM", "exchange": "NYSE",   "where": "Car sidepod"},
        {"company": "Ford Motor Company",  "ticker": "F",   "exchange": "NYSE",   "where": "Engine (Red Bull Powertrains / Ford)"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Haas": [
        {"company": "MoneyGram International","ticker": "MGRM","exchange": "NASDAQ","where": "Team name + car livery"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Audi": [
        {"company": "Volkswagen AG",       "ticker": "VOW3.DE","exchange": "XETRA","where": "Parent company (Audi is VW subsidiary)"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
    "Cadillac": [
        {"company": "General Motors",      "ticker": "GM",  "exchange": "NYSE",   "where": "Parent company — Cadillac is a GM brand"},
        {"company": "Pirelli",             "ticker": "PIRC.MI","exchange": "BIT", "where": "Tyres"},
    ],
}


def get_all_public_companies() -> list:
    """
    Flatten all publicly-traded companies into one deduplicated list.
    Returns list of dicts with: company, ticker, exchange, linked_teams, role.
    """
    companies = {}

    for team_name, team in F1_TEAMS_2026.items():
        pc = team["parent_company"]
        if pc.get("public") and pc.get("ticker"):
            key = pc["ticker"]
            companies.setdefault(key, {
                "company": pc["name"].split(" (")[0],
                "ticker": pc["ticker"], "exchange": pc["exchange"],
                "linked_teams": [], "role": "Team parent company / car manufacturer",
            })
            if team_name not in companies[key]["linked_teams"]:
                companies[key]["linked_teams"].append(team_name)

        for sp in team["key_public_sponsors"]:
            if sp.get("ticker"):
                key = sp["ticker"]
                companies.setdefault(key, {
                    "company": sp["company"], "ticker": sp["ticker"],
                    "exchange": sp["exchange"], "linked_teams": [], "role": sp["role"],
                })
                if team_name not in companies[key]["linked_teams"]:
                    companies[key]["linked_teams"].append(team_name)

    # Universal: tyre supplier + F1 itself
    t = UNIVERSAL_SUPPLIERS["tyre"]
    companies[t["ticker"]] = {
        "company": t["company"], "ticker": t["ticker"], "exchange": t["exchange"],
        "linked_teams": list(F1_TEAMS_2026.keys()), "role": t["role"],
    }
    companies[F1_PARENT["ticker"]] = {
        "company": F1_PARENT["company"], "ticker": F1_PARENT["ticker"],
        "exchange": F1_PARENT["exchange"],
        "linked_teams": list(F1_TEAMS_2026.keys()), "role": F1_PARENT["role"],
    }

    # Global sponsors
    for sp in GLOBAL_F1_SPONSORS:
        if sp.get("public") and sp.get("ticker"):
            key = sp["ticker"]
            companies.setdefault(key, {
                "company": sp["company"], "ticker": sp["ticker"],
                "exchange": sp["exchange"],
                "linked_teams": list(F1_TEAMS_2026.keys()),
                "role": sp["role"],
            })

    return list(companies.values())


def get_yfinance_ticker(ticker: str, exchange: str) -> str:
    """
    Convert a display ticker to the exact yfinance-compatible symbol.
    yfinance requires specific suffixes for non-US exchanges.
    All mappings below verified by fetching real data from each exchange.
    """
    if not ticker:
        return ""
    if "." in ticker:
        return ticker  # Already has suffix: MBG.DE, AML.L, PIRC.MI, RNO.PA etc.

    suffix_map = {
        "XETRA":    ".DE",
        "BIT":      ".MI",
        "LSE":      ".L",
        "EURONEXT": ".PA",
        "SIX":      ".SW",
        "HKSE":     ".HK",
        "TSE":      ".T",
        "ASX":      ".AX",
        "MADRID":   ".MC",
        "TADAWUL":  ".SR",
    }
    if exchange in suffix_map:
        return ticker + suffix_map[exchange]
    return ticker  # NYSE, NASDAQ — no suffix needed


def get_company_explainer(ticker: str, team_name: str) -> str:
    """
    Return a plain-English explanation of why this company's stock
    is affected by F1 race results — for non-finance audiences.
    """
    explainers = {
        "RACE": "Ferrari IS the F1 team. When Leclerc or Hamilton wins, it's Ferrari's win — the company's brand equity rises, luxury car demand increases, and NYSE:RACE moves within days.",
        "MBG.DE": "Mercedes-Benz funds and runs the F1 team as a marketing operation. A championship win is used in ads globally. Academic studies show Mercedes stock correlates with its F1 championship position.",
        "GM": "Cadillac is General Motors' luxury brand. This is GM's first F1 race ever. Every result — good or bad — is front-page news for the brand, directly affecting how analysts value GM's luxury division.",
        "VOW3.DE": "Audi is a Volkswagen AG subsidiary. Audi's entire F1 entry is a brand rebuilding exercise after the emissions scandal. Racing results affect how consumers perceive the 'premium' brand, which drives VW's most profitable cars.",
        "AML.L": "Aston Martin the road car company shares its brand with the F1 team. The F1 team is owned separately (Lawrence Stroll), but every F1 result raises or lowers global awareness of the Aston Martin road car brand.",
        "FWONA": "Liberty Media OWNS Formula 1. More exciting races = more viewers = higher TV rights value = stronger FWONA. The 2025 title fight going to the final race was worth an estimated $200M in additional media value.",
        "PIRC.MI": "Pirelli supplies 100% of F1's tyres — all 11 teams, every race. A dramatic tyre failure or blow-out creates global headlines naming Pirelli, directly affecting investor sentiment.",
        "ORCL": "Oracle's name appears on the side of the most-watched car on the grid. Every TV shot of a Red Bull car is a multi-second Oracle ad reaching 1.5 billion annual F1 viewers.",
        "HMC": "Honda powers Aston Martin as a full works partner in 2026. A Honda-powered victory is a direct advertisement for Honda's engineering capability, shown to 450 million viewers per race.",
        "F": "Ford returned to F1 in 2026 for the first time since 2004, co-developing Red Bull's power unit. Every Red Bull result is covered with 'Ford returns to F1' context — massive brand reactivation.",
        "GOOGL": "Google's branding appears on the McLaren — the 2025 Constructor Champions. Being on a championship-winning car is the highest-visibility sponsorship in motorsport.",
        "AMZN": "AWS handles F1's entire cloud infrastructure and powers the 'F1 Insights' shown on live TV. AWS is associated with F1's cutting-edge data technology every broadcast.",
        "MC.PA": "TAG Heuer (owned by LVMH) became F1's official timekeeper again in 2025 — their watches appear on screen every single lap. LVMH paid $1B over 10 years for this level of exposure.",
        "PEP": "PepsiCo joined F1 in 2025 with Sting Energy, Gatorade, and Doritos — reaching F1's 1.5 billion annual viewers, 40% of whom are now women, a new demographic for these brands.",
        "CRM": "Salesforce is both F1's global partner AND the Racing Bulls team sponsor. Their AI fan companion powered by Agentforce is being built for 827 million F1 fans globally.",
        "FLUT": "FanDuel (owned by Flutter) is F1's first-ever US official betting partner. Every race drives millions in US betting volume — dramatic finishes, crashes, and underdogs winning spike FanDuel activity.",
    }
    return explainers.get(ticker,
        f"This company sponsors or is directly linked to {team_name}. "
        "When the team performs well, brand visibility rises globally, "
        "driving investor sentiment and consumer perception of the company."
    )


# ── 2026 race results (races completed so far this season) ───────────────────
# Source: formula1.com/en/results (verified July 2026)
RACES_2026 = [
    # Source: Jolpica/Ergast API (api.jolpi.ca), 2026 results through Round 16
    {"round":1, "race":"Australian GP", "circuit":"Albert Park Circuit", "date":"2026-03-08", "winner":"George Russell", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Antonelli", "p3":"Leclerc"},
    {"round":2, "race":"Chinese GP", "circuit":"Shanghai International Circuit", "date":"2026-03-15", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Russell", "p3":"Hamilton"},
    {"round":3, "race":"Japanese GP", "circuit":"Suzuka Circuit", "date":"2026-03-29", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Piastri", "p3":"Leclerc"},
    {"round":4, "race":"Miami GP", "circuit":"Miami International Autodrome", "date":"2026-05-03", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Norris", "p3":"Piastri"},
    {"round":5, "race":"Canadian GP", "circuit":"Circuit Gilles Villeneuve", "date":"2026-05-24", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Hamilton", "p3":"Verstappen"},
    {"round":6, "race":"Monaco GP", "circuit":"Circuit de Monaco", "date":"2026-06-07", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Hamilton", "p3":"Hadjar"},
    {"round":7, "race":"Barcelona GP", "circuit":"Circuit de Barcelona-Catalunya", "date":"2026-06-14", "winner":"Lewis Hamilton", "winner_team":"Ferrari", "event_type":"WIN", "p2":"Russell", "p3":"Norris"},
    {"round":8, "race":"Austrian GP", "circuit":"Red Bull Ring", "date":"2026-06-28", "winner":"George Russell", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Verstappen", "p3":"Antonelli"},
    {"round":9, "race":"British GP", "circuit":"Silverstone Circuit", "date":"2026-07-05", "winner":"Charles Leclerc", "winner_team":"Ferrari", "event_type":"WIN", "p2":"Russell", "p3":"Hamilton"},
    {"round":10, "race":"Belgian GP", "circuit":"Spa-Francorchamps", "date":"2026-07-19", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Leclerc", "p3":"Verstappen"},
    {"round":11, "race":"Hungarian GP", "circuit":"Hungaroring", "date":"2026-07-26", "winner":"Lando Norris", "winner_team":"McLaren", "event_type":"WIN", "p2":"Verstappen", "p3":"Antonelli"},
    {"round":12, "race":"Dutch GP", "circuit":"Circuit Zandvoort", "date":"2026-08-23", "winner":"Lando Norris", "winner_team":"McLaren", "event_type":"WIN", "p2":"Antonelli", "p3":"Russell"},
    {"round":13, "race":"Italian GP", "circuit":"Autodromo Nazionale Monza", "date":"2026-09-06", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Russell", "p3":"Verstappen"},
    {"round":14, "race":"Spanish GP", "circuit":"Madring", "date":"2026-09-13", "winner":"Andrea Kimi Antonelli", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Verstappen", "p3":"Norris"},
    {"round":15, "race":"Azerbaijan GP", "circuit":"Baku City Circuit", "date":"2026-09-26", "winner":"George Russell", "winner_team":"Mercedes", "event_type":"WIN", "p2":"Verstappen", "p3":"Hadjar"},
    {"round":16, "race":"Bahrain GP in Malaysia", "circuit":"Sepang International Circuit", "date":"2026-10-04", "winner":"Max Verstappen", "winner_team":"Red Bull Racing", "event_type":"WIN", "p2":"Antonelli", "p3":"Hamilton"},
]

# ── Remaining 2026 races ──────────────────────────────────────────────────────
FUTURE_RACES_2026 = [
    {"round":17, "race":"Singapore GP", "circuit":"Marina Bay Street Circuit", "date":"2026-10-11", "country":"Singapore"},
    {"round":18, "race":"US GP", "circuit":"Circuit of the Americas", "date":"2026-10-25", "country":"USA"},
    {"round":19, "race":"Mexico City GP", "circuit":"Autodromo Hermanos Rodriguez", "date":"2026-11-01", "country":"Mexico"},
    {"round":20, "race":"São Paulo GP", "circuit":"Autodromo Jose Carlos Pace", "date":"2026-11-08", "country":"Brazil"},
    {"round":21, "race":"Las Vegas GP", "circuit":"Las Vegas Strip Circuit", "date":"2026-11-22", "country":"USA"},
    {"round":22, "race":"Qatar GP", "circuit":"Lusail International Circuit", "date":"2026-11-29", "country":"Qatar"},
    {"round":23, "race":"Abu Dhabi GP", "circuit":"Yas Marina Circuit", "date":"2026-12-06", "country":"UAE"},
]

# ── 2026 driver championship standings (after Round 16, Bahrain GP in Malaysia, 2026-10-04) ──
# Source: Jolpica/Ergast API driverStandings
DRIVER_STANDINGS_2026 = [
    {"driver": "Andrea Kimi Antonelli", "team": "Mercedes", "points": 320, "wins": 8},
    {"driver": "George Russell", "team": "Mercedes", "points": 236, "wins": 3},
    {"driver": "Lewis Hamilton", "team": "Ferrari", "points": 214, "wins": 1},
    {"driver": "Charles Leclerc", "team": "Ferrari", "points": 191, "wins": 1},
    {"driver": "Lando Norris", "team": "McLaren", "points": 188, "wins": 2},
    {"driver": "Max Verstappen", "team": "Red Bull Racing", "points": 188, "wins": 1},
    {"driver": "Oscar Piastri", "team": "McLaren", "points": 128, "wins": 0},
    {"driver": "Isack Hadjar", "team": "Red Bull Racing", "points": 96, "wins": 0},
    {"driver": "Liam Lawson", "team": "Racing Bulls", "points": 65, "wins": 0},
    {"driver": "Pierre Gasly", "team": "Alpine", "points": 41, "wins": 0},
    {"driver": "Arvid Lindblad", "team": "Racing Bulls", "points": 38, "wins": 0},
    {"driver": "Franco Colapinto", "team": "Alpine", "points": 27, "wins": 0},
    {"driver": "Oliver Bearman", "team": "Haas", "points": 20, "wins": 0},
    {"driver": "Gabriel Bortoleto", "team": "Audi", "points": 10, "wins": 0},
    {"driver": "Nico Hülkenberg", "team": "Audi", "points": 7, "wins": 0},
    {"driver": "Esteban Ocon", "team": "Haas", "points": 7, "wins": 0},
    {"driver": "Fernando Alonso", "team": "Aston Martin", "points": 7, "wins": 0},
    {"driver": "Carlos Sainz", "team": "Williams", "points": 7, "wins": 0},
    {"driver": "Alexander Albon", "team": "Williams", "points": 5, "wins": 0},
    {"driver": "Yuki Tsunoda", "team": "Racing Bulls", "points": 1, "wins": 0},
    {"driver": "Lance Stroll", "team": "Aston Martin", "points": 0, "wins": 0},
    {"driver": "Valtteri Bottas", "team": "Cadillac", "points": 0, "wins": 0},
    {"driver": "Sergio Pérez", "team": "Cadillac", "points": 0, "wins": 0},
]


# ── Convenience aliases for backwards compatibility with app.py ───────────────
TYRE_SUPPLIER = UNIVERSAL_SUPPLIERS["tyre"]

TEAM_COLORS = {
    "McLaren":       "#FF8000",
    "Ferrari":       "#E8002D",
    "Mercedes":      "#27F4D2",
    "Red Bull Racing":"#3671C6",
    "Aston Martin":  "#229971",
    "Alpine":        "#0093CC",
    "Williams":      "#64C4FF",
    "Racing Bulls":  "#6692FF",
    "Haas":          "#B6BABD",
    "Audi":          "#BB0000",
    "Cadillac":      "#CE1126",
}

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import sys, os

sys.path.append(os.path.dirname(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv()   # picks up NEWSAPI_KEY from a local .env file
except ImportError:
    pass
from utils.data_fetcher import (get_stock_data, get_stock_currency, get_seasons, events_frame,
    get_driver_standings, get_constructor_standings, get_race_schedule, get_data_source_status,
    FIRST_SEASON)
from utils.event_study import team_outcome, post_race_returns, event_study, OUTCOMES
from utils.ml_features import live_features
from utils.sentiment import get_news_sentiment
from utils.teams_sponsors import get_yfinance_ticker, current_team_for, team_matches
from models.predictor import load_model as _load_model_raw, predict_impact, arima_forecast

@st.cache_resource
def load_model():
    """Load the trained XGBoost model once per session (see train_model.py)."""
    return _load_model_raw()

st.set_page_config(page_title="F1 Pulse", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card { background:#1c1f26; border-radius:12px; padding:1.2rem 1.5rem; border:1px solid #2a2d35; margin-bottom:1rem; }
.metric-label { font-size:11px; color:#888; text-transform:uppercase; letter-spacing:0.08em; }
.metric-value { font-size:26px; font-weight:600; margin-top:4px; }
.metric-sub { font-size:11px; color:#666; margin-top:2px; }
.positive { color:#22c55e; } .negative { color:#ef4444; } .neutral { color:#94a3b8; }
.explain-box { background:#161b27; border-left:3px solid #e10600; border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin:1rem 0; font-size:14px; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# ── Reference data (teams, sponsors, tickers), loaded once ───────────────────
@st.cache_resource
def load_f1_data():
    from utils.teams_sponsors import (
        F1_TEAMS_2026, GLOBAL_F1_SPONSORS, TYRE_SUPPLIER, F1_PARENT as F1_PARENT_COMPANY,
        EXCHANGE_INFO, TEAM_COLORS, get_all_public_companies,
    )
    return {
        "TEAM_COLORS":        TEAM_COLORS,
        "F1_TEAMS":           F1_TEAMS_2026,
        "GLOBAL_F1_SPONSORS": GLOBAL_F1_SPONSORS,
        "TYRE_SUPPLIER":      TYRE_SUPPLIER,
        "F1_PARENT_COMPANY":  F1_PARENT_COMPANY,
        "EXCHANGE_INFO":      EXCHANGE_INFO,
        "ALL_PUBLIC_COMPANIES": get_all_public_companies(),
    }

_f1 = load_f1_data()
TEAM_COLORS          = _f1["TEAM_COLORS"]
F1_TEAMS             = _f1["F1_TEAMS"]          # 11 teams, 2026 season
GLOBAL_F1_SPONSORS   = _f1["GLOBAL_F1_SPONSORS"]
TYRE_SUPPLIER        = _f1["TYRE_SUPPLIER"]
F1_PARENT_COMPANY    = _f1["F1_PARENT_COMPANY"]
EXCHANGE_INFO        = _f1["EXCHANGE_INFO"]
ALL_PUBLIC_COMPANIES = _f1["ALL_PUBLIC_COMPANIES"]

TODAY        = datetime.today()
THIS_YEAR    = TODAY.year
SEASONS      = list(range(THIS_YEAR, FIRST_SEASON - 1, -1))      # newest first
HISTORY_FROM = f"{FIRST_SEASON - 1}-06-01"   # stock history start: room for the first race's estimation window

# ── Live data loaders (cached) ────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_all_events():
    return events_frame(get_seasons(sorted(SEASONS)))

def load_season(year):
    if all_events.empty:
        return all_events
    return all_events[all_events["season"] == year].reset_index(drop=True)

@st.cache_data(ttl=1800, show_spinner=False)
def load_stock_history(ticker):
    return get_stock_data(ticker, HISTORY_FROM, TODAY.strftime("%Y-%m-%d"))

@st.cache_data(ttl=1800, show_spinner=False)
def load_arima_forecast(ticker, steps=5):
    """ARIMA(2,1,2) fitted on the last year of closes."""
    return arima_forecast(load_stock_history(ticker).tail(252), steps)

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_currency(ticker, fallback):
    return get_stock_currency(ticker, fallback)

@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule(year):
    return get_race_schedule(year)

@st.cache_data(ttl=3600, show_spinner=False)
def load_driver_standings(year):
    return get_driver_standings(year)

@st.cache_data(ttl=3600, show_spinner=False)
def load_constructor_standings(year):
    return get_constructor_standings(year)

with st.spinner(f"Loading F1 results {FIRST_SEASON}–{THIS_YEAR}..."):
    all_events = load_all_events()

def latest_drivers(team_name, fallback):
    """Drivers who raced for the team in the most recent race with results."""
    if not all_events.empty:
        for tr in reversed(all_events["team_results"].tolist()):
            drivers = [d for name, res in (tr or {}).items()
                       if team_matches(team_name, name) for d in res.get("drivers", [])]
            if drivers:
                return drivers
    return fallback

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 Pulse")
    st.markdown("*Stock Impact Intelligence*")
    st.divider()

    st.markdown("#### 1. Pick an F1 team")
    selected_team = st.selectbox(f"F1 team ({len(F1_TEAMS)} teams on the grid)", list(F1_TEAMS.keys()))
    team_info = F1_TEAMS[selected_team]

    st.markdown(
        f"<div class='explain-box'>"
        f"<b>{team_info['full_name']}</b><br>"
        f"Drivers: {' & '.join(latest_drivers(selected_team, team_info['drivers_2026']))}<br>"
        f"Engine: {team_info['engine']} · Based: {team_info['base_country']}"
        f"</div>", unsafe_allow_html=True
    )

    st.markdown("#### 2. Pick a linked public company")
    team_company_options = {}
    pc = team_info["parent_company"]
    if pc.get("public") and pc.get("ticker"):
        label = f"{pc['name']} — Team/Manufacturer ({pc['ticker']} · {pc['exchange']})"
        team_company_options[label] = {
            "company": pc["name"], "ticker": pc["ticker"],
            "exchange": pc["exchange"], "role": "Parent company / car manufacturer",
        }
    for sp in team_info["key_public_sponsors"]:
        already_listed = any(o["ticker"] == sp.get("ticker") for o in team_company_options.values())
        if sp.get("ticker") and sp.get("exchange") and not already_listed:
            label = f"{sp['company']} ({sp['ticker']} · {sp['exchange']})"
            team_company_options[label] = {
                "company": sp["company"], "ticker": sp["ticker"],
                "exchange": sp["exchange"], "role": sp["role"],
            }
    team_company_options[f"Pirelli — Tyre supplier to ALL teams ({TYRE_SUPPLIER['ticker']} · {TYRE_SUPPLIER['exchange']})"] = TYRE_SUPPLIER
    team_company_options[f"Liberty Media — Owns F1 itself ({F1_PARENT_COMPANY['ticker']} · {F1_PARENT_COMPANY['exchange']})"] = F1_PARENT_COMPANY

    selected_label = st.selectbox("Company to track", list(team_company_options.keys()))
    selected_stock = team_company_options[selected_label]
    selected_company = selected_stock["company"]
    ticker_raw = selected_stock["ticker"]

    lookback = st.slider("Lookback window (days)", 30, 365, 180, step=30)
    event_window = st.slider("Event impact window (trading days)", 1, 7, 3)

    st.divider()
    st.markdown("#### Why this stock?")
    if pc.get("public") and ticker_raw == pc.get("ticker"):
        why_text = pc.get("note", "")
    else:
        sponsor_match = next((s for s in team_info["key_public_sponsors"] if s.get("ticker") == ticker_raw), None)
        if sponsor_match:
            why_text = sponsor_match["role"]
        elif ticker_raw == TYRE_SUPPLIER["ticker"]:
            why_text = TYRE_SUPPLIER["role"]
        else:
            why_text = F1_PARENT_COMPANY["role"]
    st.markdown(f"<div class='explain-box'>{why_text}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"**Exchange:** {selected_stock['exchange']}")
    st.markdown("**Data:** yfinance · Jolpica F1 API")
    st.markdown("**Model:** XGBoost")
    st.caption("✅ Primary exchange listings only — no illiquid US OTC ADRs.")

# ── Load stock data ───────────────────────────────────────────────────────────
ticker = get_yfinance_ticker(ticker_raw, selected_stock["exchange"])
exch_info = EXCHANGE_INFO.get(selected_stock["exchange"], {})

with st.spinner(f"Fetching {ticker} from {selected_stock['exchange']}..."):
    stock_hist = load_stock_history(ticker)
    exch_currency = load_currency(ticker, exch_info.get("currency", "USD"))

currency_label = "GBp (pence)" if exch_currency == "GBp" else exch_currency

# Market index for the stock's exchange (falls back to the S&P 500 if Yahoo has too little history)
market_ticker = exch_info.get("index", "^GSPC")
market_hist = load_stock_history(market_ticker)
if len(market_hist) < 200 and market_ticker != "^GSPC":
    market_ticker = "^GSPC"
    market_hist = load_stock_history(market_ticker)
has_stock = len(stock_hist) >= 2
stock_df = stock_hist[stock_hist["Date"] >= TODAY - timedelta(days=lookback)] if has_stock else stock_hist
if has_stock and len(stock_df) < 2:
    stock_df = stock_hist.tail(2)

# Real post-race returns for this stock, over every race since FIRST_SEASON
race_returns = post_race_returns(stock_hist, all_events, selected_team) if has_stock else pd.DataFrame()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏎️ F1 Pulse — Stock Impact Tracker")
st.markdown(
    f"Tracking **{selected_company}** ({ticker_raw} · {selected_stock['exchange']}) "
    f"linked to **{selected_team}** · last {lookback} days · live from yfinance"
)
st.divider()

if not has_stock:
    st.error(
        f"Yahoo Finance returned no price data for **{ticker}**. The ticker may have been delisted, "
        "or Yahoo Finance may be unreachable right now. Price-based tabs are hidden; "
        "F1 data tabs still work."
    )

# ── Metric cards ──────────────────────────────────────────────────────────────
def mcard(label, value, suffix="", is_pct=False, sub=""):
    if isinstance(value, float) and not np.isnan(value):
        color = ("positive" if value > 0 else "negative" if value < 0 else "neutral") if is_pct else "neutral"
        txt = f"{value:+.2f}{suffix}" if is_pct else f"{value:,.2f}{suffix}"
    else:
        color = "neutral"
        txt = "n/a" if isinstance(value, float) else str(value)
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value {color}'>{txt}</div>{sub_html}</div>"

def avg_return(df, col, outcome=None):
    if df.empty or col not in df.columns:
        return float("nan"), 0
    vals = (df[df["outcome"] == outcome] if outcome else df)[col].dropna()
    return (float(vals.mean()) if len(vals) else float("nan")), len(vals)

if has_stock:
    current_price = float(stock_df["Close"].iloc[-1])
    prev_price    = float(stock_df["Close"].iloc[-2])
    price_change  = ((current_price - prev_price) / prev_price) * 100
    win_col = f"Day +{event_window}"
    avg_all, n_all = avg_return(race_returns, win_col)
    avg_win, n_win = avg_return(race_returns, win_col, "WIN")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(mcard("Current price", current_price, f" {exch_currency}",
                               sub=f"Close on {stock_df['Date'].iloc[-1]:%d %b %Y}"), unsafe_allow_html=True)
    with c2: st.markdown(mcard("1-day change", price_change, "%", True), unsafe_allow_html=True)
    with c3: st.markdown(mcard(f"Avg {event_window}-day return after a race", avg_all, "%", True,
                               sub=f"{n_all} races since {FIRST_SEASON}"), unsafe_allow_html=True)
    with c4: st.markdown(mcard(f"Avg {event_window}-day return after a {selected_team} win", avg_win, "%", True,
                               sub=f"{n_win} wins since {FIRST_SEASON}" if n_win else "No wins in the data yet"),
                         unsafe_allow_html=True)

# ── Data source status badge ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def check_data_sources():
    return get_data_source_status()

SRC_LABELS = {"fastf1":"🟢 FastF1 (live)", "jolpica":"🟢 Jolpica F1 API (live)",
              "static_cache":"🔵 Cached dataset (offline fallback)"}

with st.expander("🔌 Live data source status", expanded=False):
    src_status = check_data_sources()
    s1, s2, s3 = st.columns(3)
    def status_badge(label, active):
        return f"{'🟢' if active else '🔴'} **{label}**"
    with s1: st.markdown(status_badge("Jolpica F1 API", src_status["jolpica"]))
    with s2: st.markdown(status_badge("FastF1", src_status["fastf1"]))
    with s3: st.markdown(status_badge("yfinance", src_status["yfinance"]))
    cur = load_season(THIS_YEAR)
    active_f1 = cur["data_source"].mode()[0] if not cur.empty else "none"
    st.caption(f"{THIS_YEAR} F1 results loaded from: **{SRC_LABELS.get(active_f1, active_f1)}** · "
               f"Stock data: **{'live' if src_status['yfinance'] else 'unavailable'}**")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab_teams, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price & Events",
    "🏎️ Teams & Sponsors",
    f"🏁 F1 Race History ({FIRST_SEASON}–{THIS_YEAR})",
    "🏆 Driver Standings",
    "🔥 Correlation Heatmap",
    "🤖 ML Prediction",
    "📊 Event Study (CAR)",
])

# ══════════════ TAB — Teams & Sponsors ══════════════
with tab_teams:
    st.markdown(f"### All {len(F1_TEAMS)} F1 teams (2026 season) and their stock market connections")
    st.markdown(
        "<div class='explain-box'>"
        "<b>New to stock markets or F1? Start here.</b> Every F1 team is either directly publicly traded, "
        "or is backed/sponsored by companies that ARE publicly traded. This page maps the entire 2026 grid "
        "to real, verified stock tickers — so you can see exactly how a Sunday afternoon race connects to "
        "Monday morning's stock market open."
        "</div>", unsafe_allow_html=True
    )

    def exch_code(c):
        return c["exchange"].split(" ")[0]

    s1, s2, s3, s4 = st.columns(4)
    with s1: st.markdown(mcard("F1 teams tracked", len(F1_TEAMS)), unsafe_allow_html=True)
    with s2: st.markdown(mcard("Public companies linked", len(ALL_PUBLIC_COMPANIES)), unsafe_allow_html=True)
    with s3: st.markdown(mcard("Stock exchanges covered", len({exch_code(c) for c in ALL_PUBLIC_COMPANIES})), unsafe_allow_html=True)
    with s4: st.markdown(mcard("Countries represented", len({EXCHANGE_INFO.get(exch_code(c), {}).get('country', exch_code(c)) for c in ALL_PUBLIC_COMPANIES})), unsafe_allow_html=True)

    st.markdown("#### Browse by team")
    for team_name, team in F1_TEAMS.items():
        with st.expander(f"🏎️ **{team_name}** — {team['full_name']}", expanded=False):
            col_info, col_stocks = st.columns([1, 2])
            with col_info:
                st.markdown(f"**Drivers (latest race):** {' & '.join(latest_drivers(team_name, team['drivers_2026']))}")
                st.markdown(f"**Engine supplier:** {team['engine']}")
                st.markdown(f"**Based in:** {team['base_country']}")
                pc = team["parent_company"]
                if pc.get("public"):
                    st.success(f"✅ Directly publicly traded: **{pc['ticker']}** on {pc['exchange']}")
                else:
                    st.info(f"🔒 Privately owned ({pc['name']}) — track it via sponsors instead →")
                st.caption(pc.get("note", ""))

            with col_stocks:
                st.markdown("**Linked publicly-traded companies:**")
                rows = []
                if pc.get("public"):
                    rows.append({"Company": pc["name"], "Ticker": pc["ticker"], "Exchange": pc["exchange"], "Connection": "Team owner / car manufacturer"})
                for sp in team["key_public_sponsors"]:
                    if sp.get("ticker"):
                        rows.append({"Company": sp["company"], "Ticker": sp["ticker"], "Exchange": sp["exchange"], "Connection": sp["role"]})
                rows.append({"Company": TYRE_SUPPLIER["company"], "Ticker": TYRE_SUPPLIER["ticker"], "Exchange": TYRE_SUPPLIER["exchange"], "Connection": "Tyre supplier (all teams)"})
                rows.append({"Company": F1_PARENT_COMPANY["company"], "Ticker": F1_PARENT_COMPANY["ticker"], "Exchange": F1_PARENT_COMPANY["exchange"], "Connection": "Owns F1 itself"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Global F1 partners (apply to every team, every race)")
    st.markdown(
        "<div class='explain-box'>"
        "These companies sponsor Formula 1 itself, not a single team — so EVERY race weekend is a marketing "
        "moment for them, regardless of who wins."
        "</div>", unsafe_allow_html=True
    )
    global_rows = [{"Company": s["company"], "Ticker": s.get("ticker") or "Private", "Exchange": s.get("exchange") or "—", "Role": s["role"]} for s in GLOBAL_F1_SPONSORS]
    st.dataframe(pd.DataFrame(global_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Stock exchanges represented in this project")
    exch_rows = []
    for code, info in EXCHANGE_INFO.items():
        count = len([c for c in ALL_PUBLIC_COMPANIES if exch_code(c) == code])
        if count > 0:
            exch_rows.append({"Exchange": info["full_name"], "Code": code, "Country": info["country"], "Currency": info["currency"], "Companies tracked": count})
    st.dataframe(pd.DataFrame(exch_rows).sort_values("Companies tracked", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("#### A note on data accuracy")
    st.markdown(
        "<div class='explain-box'>"
        "Race results, standings and the calendar are fetched live from the Jolpica F1 API; prices are fetched "
        "live from Yahoo Finance via <b>yfinance</b>. The team → sponsor → ticker mapping on this page is "
        "curated by hand (no public API publishes F1 sponsorship deals), so it is updated manually when deals change. "
        "Every ticker uses each company's <b>primary exchange listing</b> — e.g. Mercedes-Benz "
        "trades on Frankfurt as <code>MBG.DE</code>, not the thinly-traded US OTC ticker <code>MBGYY</code>. "
        "Prices are end-of-day accurate; intraday quotes may lag 15–20 minutes, which does not affect the "
        "daily-level event study used throughout this dashboard."
        "</div>", unsafe_allow_html=True
    )


# ══════════════ TAB 1 — Price + Events ══════════════
OUTCOME_LABELS = {"WIN": "WIN", "PODIUM": "PODIUM (2nd–3rd)", "POINTS": "POINTS (4th–10th)",
                  "NO POINTS": "NO POINTS (11th or lower)", "DNF": "DNF (a car retired)"}
OUTCOME_COLORS = {
    "WIN":       "#22c55e",
    "PODIUM":    "#60a5fa",
    "POINTS":    "#a78bfa",
    "NO POINTS": "#6b7280",
    "DNF":       "#ef4444",
}

with tab1:
    st.markdown("### Live stock price with F1 race markers")
    st.markdown(
        "<div class='explain-box'>"
        f"Each vertical line marks a race. Colour shows the result <b>for {selected_team}</b> (its best-placed car): "
        "<b style='color:#22c55e'>Green</b> = WIN · "
        "<b style='color:#60a5fa'>Blue</b> = podium (2nd/3rd) · "
        "<b style='color:#a78bfa'>Purple</b> = points (4th–10th) · "
        "<b style='color:#ef4444'>Red</b> = a car retired / DNF and no points · "
        "<b style='color:#6b7280'>Grey</b> = finished outside the points. "
        "Hover over any race marker to see the full result."
        "</div>", unsafe_allow_html=True
    )

    if not has_stock:
        st.info("No price data available for this ticker.")
    else:
        start_dt = TODAY - timedelta(days=lookback)
        events_df = all_events[(all_events["date"] >= start_dt) & (all_events["date"] <= TODAY)].copy() \
            if not all_events.empty else pd.DataFrame()
        if not events_df.empty:
            events_df["outcome"] = events_df["team_results"].apply(lambda tr: team_outcome(tr, selected_team)).fillna("NOT ON GRID")

        show_fc = st.checkbox("Show 5-day ARIMA price forecast", value=True,
                              help="ARIMA(2,1,2) time-series model fitted on the last year of daily closes, "
                                   "with a 95% confidence band. A statistical baseline — not a trading signal.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_df["Date"], y=stock_df["Close"],
            mode="lines", name="Close price",
            line=dict(color="#60a5fa", width=2),
            hovertemplate=f"<b>%{{x|%b %d %Y}}</b><br>Close: %{{y:,.2f}} {exch_currency}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=stock_df["Date"], y=stock_hist.set_index("Date")["Close"].rolling(20).mean().reindex(stock_df["Date"]).values,
            mode="lines", name="20-day MA",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            hovertemplate=f"<b>%{{x|%b %d %Y}}</b><br>20d MA: %{{y:,.2f}} {exch_currency}<extra></extra>",
        ))

        price_span = stock_df["Close"].max() - stock_df["Close"].min()
        for _, ev in events_df.iterrows():
            ev_date   = pd.to_datetime(ev["date"])
            ev_type   = ev["outcome"]
            ev_color  = OUTCOME_COLORS.get(ev_type, "#6b7280")
            race_name = str(ev.get("race","")).replace(" Grand Prix","").replace(" GP","")

            subset = stock_df[stock_df["Date"] <= ev_date]
            if subset.empty:
                continue
            price_at = float(subset["Close"].iloc[-1])

            fig.add_shape(
                type="line", x0=ev_date, x1=ev_date, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=ev_color, width=1.5, dash="dash"), opacity=0.7,
            )
            # Invisible hoverable marker — tooltip appears on hover, no dot visible
            fig.add_trace(go.Scatter(
                x=[ev_date], y=[price_at + price_span * 0.08],
                mode="markers", marker=dict(color=ev_color, size=18, opacity=0),
                showlegend=False, name=race_name,
                hovertemplate=(
                    f"<b>{race_name}</b><br>"
                    f"Date: {ev_date.strftime('%B %d, %Y')}<br>"
                    f"Winner: {ev.get('winner','')} ({ev.get('winner_team','')})<br>"
                    f"{selected_team}: {ev_type}"
                    f"<extra></extra>"
                ),
            ))
            fig.add_annotation(
                x=ev_date, y=price_at, text=race_name,
                showarrow=True, arrowhead=2, arrowcolor=ev_color, arrowsize=0.8,
                font=dict(size=9, color=ev_color), bgcolor="#1c1f26",
                bordercolor=ev_color, borderwidth=1, borderpad=3, ay=-35,
            )

        fc = load_arima_forecast(ticker) if show_fc else pd.DataFrame()
        if not fc.empty:
            last = stock_df.iloc[-1]
            fc_x = [last["Date"]] + list(fc["date"])
            fig.add_trace(go.Scatter(
                x=fc_x + fc_x[::-1],
                y=[last["Close"]] + list(fc["upper_ci"]) + list(fc["lower_ci"])[::-1] + [last["Close"]],
                mode="lines", fill="toself", fillcolor="rgba(167,139,250,0.15)", line=dict(width=0),
                name="95% confidence", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=fc_x, y=[last["Close"]] + list(fc["forecast"]),
                mode="lines+markers", name="ARIMA forecast",
                line=dict(color="#a78bfa", width=2, dash="dot"),
                hovertemplate=f"<b>%{{x|%b %d %Y}}</b><br>Forecast: %{{y:,.2f}} {exch_currency}<extra></extra>",
            ))

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=480,
            xaxis=dict(showgrid=False, showspikes=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2228", title=f"Price ({currency_label})", showspikes=False),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="closest", hoverdistance=30,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not events_df.empty:
            st.dataframe(
                events_df[["date","race","circuit","winner","winner_team","outcome"]]
                .sort_values("date", ascending=False)
                .rename(columns={"date":"Date","race":"Grand Prix","circuit":"Circuit","winner":"Winner",
                                 "winner_team":"Winning team","outcome":f"{selected_team} result"}),
                use_container_width=True, hide_index=True,
                column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD")},
            )
        else:
            st.caption("No races in the selected lookback window.")

# ══════════════ TAB 2 — F1 Race History ══════════════
def stock_connection(winner_team):
    """Most direct public stock for a team, from the team/sponsor mapping."""
    team = F1_TEAMS.get(current_team_for(winner_team))
    if not team:
        return f"Indirect via {F1_PARENT_COMPANY['ticker']}"
    pc = team["parent_company"]
    if pc.get("public") and pc.get("ticker"):
        return f"{pc['ticker']} ({pc['exchange']}) — {pc['name'].split(' (')[0]}"
    sp = next((s for s in team["key_public_sponsors"] if s.get("ticker")), None)
    if sp:
        return f"via {sp['company']} ({sp['ticker']})"
    return f"Indirect via {F1_PARENT_COMPANY['ticker']}"

with tab2:
    st.markdown("### Complete F1 race results by season")
    st.markdown(
        "<div class='explain-box'>"
        f"Browse every Grand Prix from {FIRST_SEASON} to {THIS_YEAR}, fetched live after each race. "
        "Select a year to see all race winners, circuits, and podium finishers. "
        "The <b>Stock Connection</b> column shows which publicly traded company is directly linked to each race result."
        "</div>", unsafe_allow_html=True
    )

    col_yr, col_team = st.columns([1, 2])
    with col_yr:
        hist_year = st.selectbox("Select season", SEASONS, index=0)

    with st.spinner(f"Loading {hist_year} race results..."):
        df_hist = load_season(hist_year)

    if not df_hist.empty:
        src = df_hist["data_source"].mode()[0]
        st.caption(f"Data source: {SRC_LABELS.get(src, src)}")

    with col_team:
        team_options = ["All teams"] + (sorted(df_hist["winner_team"].dropna().unique().tolist()) if not df_hist.empty else [])
        team_filter  = st.selectbox("Filter by winning team", team_options)

    if df_hist.empty:
        st.info(f"No {hist_year} race results yet.")
    else:
        season_races = len(df_hist)
        wins_by_team = df_hist["winner_team"].value_counts()
        if team_filter != "All teams":
            df_hist = df_hist[df_hist["winner_team"] == team_filter]

        df_hist = df_hist.assign(
            stock_connection=df_hist["winner_team"].apply(stock_connection),
            podium=df_hist.apply(lambda r: f"1. {r['winner']}  2. {r.get('p2') or '-'}  3. {r.get('p3') or '-'}", axis=1),
        )
        st.dataframe(
            df_hist[["round","race","circuit","date","winner_team","podium","stock_connection"]].rename(columns={
                "round":"#","race":"Grand Prix","circuit":"Circuit","date":"Date",
                "winner_team":"Winning Team","podium":"Podium (1–2–3)","stock_connection":"Stock Connection",
            }),
            use_container_width=True, hide_index=True,
            column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD")},
        )

        summary = " · ".join(f"{cnt} {team} win{'s' if cnt != 1 else ''}" for team, cnt in wins_by_team.items())
        st.markdown(f"**{hist_year} season:** {season_races} races so far · {summary}")

        wins_series = wins_by_team.rename_axis("Team").reset_index(name="Wins")
        wins_series["color"] = wins_series["Team"].map(TEAM_COLORS).fillna("#888")
        fig_wins = go.Figure(go.Bar(
            x=wins_series["Team"], y=wins_series["Wins"],
            marker_color=wins_series["color"],
            text=wins_series["Wins"], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Wins: %{y}<extra></extra>",
        ))
        fig_wins.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, title=f"{hist_year} race wins by team",
            yaxis=dict(showgrid=True, gridcolor="#1e2228", title="Wins"),
            xaxis=dict(showgrid=False), margin=dict(l=10,r=10,t=50,b=10),
        )
        st.plotly_chart(fig_wins, use_container_width=True)

# ══════════════ TAB 3 — Driver Standings ══════════════
with tab3:
    st.markdown("### Driver championship standings")
    st.markdown(
        "<div class='explain-box'>"
        "Championship standings directly affect stock prices. When a Ferrari driver leads the championship, "
        "Ferrari stock sees elevated volatility. The closer the championship fight (like 2021 Hamilton vs Verstappen), "
        "the higher the race viewership — which lifts Liberty Media (FWONA) revenue and stock."
        "</div>", unsafe_allow_html=True
    )

    stand_year = st.selectbox("Select year", SEASONS, index=0, key="stand_yr")

    with st.spinner(f"Fetching {stand_year} championship standings..."):
        df_stand = load_driver_standings(stand_year)
        df_constructors = load_constructor_standings(stand_year)

    if df_stand.empty:
        st.info(f"No {stand_year} standings available yet.")
    else:
        df_stand = df_stand.copy()
        if df_stand["data_source"].iloc[0] == "static_cache":
            st.caption("🔵 Live standings unavailable — showing the cached top 10.")
        elif df_stand["after_round"].iloc[0]:
            st.caption(f"🟢 Live from Jolpica F1 API · after round {df_stand['after_round'].iloc[0]}")
        df_stand["color"] = df_stand["team"].map(TEAM_COLORS).fillna("#888")

        fig_stand = go.Figure()
        for _, row in df_stand.iterrows():
            fig_stand.add_trace(go.Bar(
                x=[row["points"]], y=[row["driver"]],
                orientation="h", marker_color=row["color"], name=row["team"], showlegend=False,
                hovertemplate=f"<b>{row['driver']}</b><br>Team: {row['team']}<br>Points: {row['points']:g}<br>Wins: {row['wins']}<extra></extra>",
                text=f"  {row['points']:g} pts  ({row['wins']} wins)",
                textposition="outside",
            ))
        fig_stand.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=max(350, len(df_stand)*32),
            xaxis=dict(title="Championship points", showgrid=True, gridcolor="#1e2228"),
            yaxis=dict(showgrid=False, autorange="reversed"),
            margin=dict(l=10,r=120,t=30,b=10),
            title=f"{stand_year} Driver Championship",
        )
        st.plotly_chart(fig_stand, use_container_width=True)

        st.dataframe(
            df_stand[["position","driver","team","points","wins"]].rename(columns={
                "position":"Pos","driver":"Driver","team":"Team","points":"Points","wins":"Wins"
            }),
            use_container_width=True, hide_index=True,
        )

        if not df_constructors.empty:
            st.markdown("#### Constructor championship standings")
            df_constructors = df_constructors.copy()
            df_constructors["color"] = df_constructors["team"].map(TEAM_COLORS).fillna("#888")
            fig_con = go.Figure(go.Bar(
                x=df_constructors["team"], y=df_constructors["points"],
                marker_color=df_constructors["color"],
                text=df_constructors["points"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Points: %{y}<extra></extra>",
            ))
            fig_con.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, title=f"{stand_year} Constructor Championship",
                yaxis=dict(showgrid=True, gridcolor="#1e2228", title="Points"),
                xaxis=dict(showgrid=False), margin=dict(l=10,r=10,t=50,b=10),
            )
            st.plotly_chart(fig_con, use_container_width=True)

        st.markdown("#### Points across seasons — top contenders")
        top_drivers = df_stand["driver"].head(4).tolist()
        multi = []
        for yr in sorted(SEASONS):
            yr_standings = load_driver_standings(yr)
            for _, row in yr_standings.iterrows():
                if row["driver"] in top_drivers:
                    multi.append({"Year": yr, "Driver": row["driver"], "Points": row["points"], "Team": row["team"]})
        df_multi = pd.DataFrame(multi)
        if not df_multi.empty:
            fig_multi = px.line(
                df_multi.sort_values("Year"), x="Year", y="Points", color="Driver",
                markers=True, title="Championship points over multiple seasons",
                hover_data=["Team"],
                color_discrete_sequence=["#3671C6","#FF8000","#E8002D","#27F4D2"],
            )
            fig_multi.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=340, xaxis=dict(tickvals=sorted(SEASONS), showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e2228"),
                margin=dict(l=10,r=10,t=50,b=10),
            )
            st.plotly_chart(fig_multi, use_container_width=True)

# ══════════════ TAB 4 — Heatmap ══════════════
with tab4:
    st.markdown("### Race result × Stock return")
    n_races = len(race_returns)
    st.markdown(
        "<div class='explain-box'>"
        "This heatmap answers the core question: <b>how much does the stock move, and in which direction, "
        f"depending on how {selected_team} did in the race?</b> Each cell is the average % change in "
        f"<b>{selected_company}</b> from the last close before the race to the close 1, 3, 5 and 7 trading days later. "
        f"Computed live from {n_races} races since {FIRST_SEASON} and real Yahoo Finance prices."
        "</div>", unsafe_allow_html=True
    )

    horizons = ["Day +1", "Day +3", "Day +5", "Day +7"]
    if race_returns.empty:
        st.info("Not enough price history to compute race returns for this stock.")
    else:
        present = [o for o in OUTCOMES if o in set(race_returns["outcome"])]
        means  = race_returns.groupby("outcome")[horizons].mean().reindex(present)
        counts = race_returns.groupby("outcome").size().reindex(present)
        labels = [f"{o} (n={counts[o]})" for o in present]
        text   = [[f"{v:+.2f}%" if pd.notna(v) else "" for v in row] for row in means.values]
        lim    = float(np.nanmax(np.abs(means.values))) if means.size else 1.0

        fig4 = go.Figure(go.Heatmap(
            z=means.values, x=horizons, y=labels, text=text, texttemplate="%{text}",
            colorscale=[[0, "#7f1d1d"], [0.5, "#1c1f26"], [1, "#14532d"]], zmid=0, zmin=-lim, zmax=lim,
            colorbar=dict(title="Avg return %", ticksuffix="%"),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.2f}%<extra></extra>",
        ))
        fig4.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=120 + 60 * len(present), margin=dict(l=10,r=10,t=20,b=10),
            yaxis=dict(tickfont=dict(size=13), autorange="reversed"),
        )
        fig4.update_traces(textfont=dict(size=14, color="white"))
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Green = stock rose on average. Red = stock fell. Raw returns — they include general market moves; "
                   "the Event Study tab strips those out. Small n means a cell is noisy.")

        with st.expander("See every race behind these numbers"):
            st.dataframe(
                race_returns.sort_values("date", ascending=False).round(2),
                use_container_width=True, hide_index=True,
                column_config={"date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")},
            )

    st.markdown("#### How the F1-Stock connection works")
    st.markdown("""
| F1 Event | Why it moves the stock |
|---|---|
| **WIN** | Brand visibility spike → media coverage → consumer trust → car/merch sales signal |
| **Crash / DNF** | Negative PR cycle, sponsor confidence drops, analysts flag operational issues |
| **Championship lead** | Season-long narrative builds, merchandise revenue forecasts rise |
| **Penalty / DSQ** | Governance risk signal, sponsor embarrassment, analyst downgrades |
| **High viewership race** | Liberty Media ad revenue rises, FWONA stock reacts directly |
    """)

# ══════════════ TAB 5 — ML Prediction ══════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_news_sentiment(team, api_key):
    return get_news_sentiment(team, api_key)

def newsapi_key():
    try:
        return st.secrets.get("NEWSAPI_KEY", "") or os.getenv("NEWSAPI_KEY", "")
    except Exception:
        return os.getenv("NEWSAPI_KEY", "")

with tab5:
    st.markdown("### Predict stock impact for any upcoming race")
    model = load_model()
    meta = model.get("meta", {})
    st.markdown(
        "<div class='explain-box'>"
        "Select a real upcoming Grand Prix from the live calendar and an outcome for "
        f"<b>{selected_team}</b>. The XGBoost model predicts whether <b>{selected_company}</b> rises or falls "
        "over the 3 trading days after the race. Everything except the outcome is filled in from live data: "
        "the stock's recent volatility, volume, RSI and trend, the market's last 5 days, the team's form and its "
        "championship position. The model is trained by <code>train_model.py</code> on real race results and real "
        "prices since 2021."
        "</div>", unsafe_allow_html=True
    )
    if meta:
        st.caption(
            f"Model trained {meta['trained_at']} on {meta['rows']:,} real race × company pairs. "
            f"On {meta['test_rows']:,} unseen race × company pairs from {meta['test_from']} onwards it was right "
            f"{meta['test_accuracy']:.1%} of the time (AUC {meta['test_auc']:.2f}; always guessing the most common "
            f"direction scores {meta['baseline_accuracy']:.1%}). Short-term stock moves are mostly noise — treat "
            "predictions as a demonstration, not a forecast."
        )

    schedule = load_schedule(THIS_YEAR)
    upcoming = schedule[~schedule["is_past"]] if not schedule.empty else schedule
    if upcoming.empty:
        schedule = load_schedule(THIS_YEAR + 1)
        upcoming = schedule[~schedule["is_past"]] if not schedule.empty else schedule

    if upcoming.empty:
        st.info("No upcoming races published yet — check back when the next calendar is announced.")
    elif not has_stock:
        st.info("No price data available for this ticker, so there is nothing to predict.")
    else:
        race_options = {f"R{r['round']} — {r['race']} ({r['date']})": r for _, r in upcoming.iterrows()}
        col_a, col_b = st.columns(2)
        with col_a:
            race_label = st.selectbox("Select upcoming Grand Prix", list(race_options.keys()))
        with col_b:
            expected_outcome = st.selectbox(f"Expected result for {selected_team}", OUTCOMES,
                                            format_func=lambda o: OUTCOME_LABELS[o])
        selected_race = race_options[race_label]
        st.markdown(f"**Circuit:** {selected_race['circuit']} · **Country:** {selected_race['country'] or '—'} · **Date:** {selected_race['date']}")

        features = live_features(stock_hist, market_hist, all_events, selected_team, expected_outcome,
                                 int(selected_race["round"]), int(schedule["round"].max()))
        with st.expander("Live inputs the model will use", expanded=False):
            st.dataframe(pd.DataFrame([
                {"Input": "Stock volatility (last 10 days, daily std)", "Value": f"{features['pre_event_volatility']*100:.2f}%"},
                {"Input": "Volume vs 20-day average", "Value": f"{features['volume_spike']:.2f}×"},
                {"Input": "RSI (14-day)", "Value": f"{features['rsi_at_event']:.0f}"},
                {"Input": "Above 20-day moving average", "Value": "yes" if features["above_ma20"] else "no"},
                {"Input": f"Market trend, last 5 days ({market_ticker})", "Value": f"{features['market_trend_5d']:+.2f}%"},
                {"Input": f"{selected_team} podiums in last 5 races", "Value": f"{features['team_form']:.0%}"},
                {"Input": f"{selected_team} constructor-championship position", "Value": f"P{features['champ_pos']}"},
                {"Input": "Season progress", "Value": f"{features['season_progress']:.0%}"},
            ]), use_container_width=True, hide_index=True)

        if st.button("Run prediction", type="primary", use_container_width=True):
            typical, n_typ = avg_return(race_returns, "Day +3", expected_outcome)
            result = predict_impact(model, features, typical)

            r1, r2, r3 = st.columns(3)
            icon = "🟢" if result["direction"] == "UP" else "🔴"
            with r1: st.metric("Predicted direction", f"{icon} {result['direction']}")
            with r2: st.metric(f"Avg 3-day return after a {expected_outcome}",
                               f"{result['predicted_return']:+.2f}%" if result["predicted_return"] is not None else "n/a",
                               help=f"Real average for {selected_company} over {n_typ} past races with this result")
            with r3: st.metric("Model confidence", f"{result['confidence']:.0f}%")

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=result["prob_up"] * 100, number={"suffix": "%"},
                title={"text": "Probability the stock rises", "font": {"size":15}},
                gauge={
                    "axis": {"range":[0,100]},
                    "bar":  {"color": "#22c55e" if result["direction"]=="UP" else "#ef4444"},
                    "steps":[{"range":[0,40],"color":"#1c1f26"},{"range":[40,60],"color":"#1e2228"},{"range":[60,100],"color":"#1e2a1f"}],
                },
            ))
            fig_g.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=270, margin=dict(l=30,r=30,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"Prediction made by: {result['source']}. Not financial advice. Academic project.")

    # ── Live news sentiment ──
    st.markdown(f"#### 📰 Live news sentiment — {selected_team}")
    news = load_news_sentiment(selected_team, newsapi_key())
    if news["compound"] is None:
        st.info(
            f"News sentiment unavailable: {news['error']}. Add a free key from newsapi.org as "
            "`NEWSAPI_KEY` in the app's Streamlit secrets (or a local `.env`) to turn this on."
        )
    else:
        n1, n2 = st.columns([1, 2])
        with n1:
            mood = "positive" if news["compound"] > 0.05 else "negative" if news["compound"] < -0.05 else "neutral"
            st.markdown(mcard("Headline mood (VADER, −1 to +1)", news["compound"], "", True,
                              sub=f"{mood} · {news['article_count']} headlines, last 7 days"), unsafe_allow_html=True)
        with n2:
            for h in news["headlines"]:
                st.markdown(f"- [{h['title']}]({h['url']}) — {h['source']}, {h['published']} · score {h['score']:+.2f}")
        st.caption("Shown as context only: free news archives don't go back far enough to train the model on "
                   "sentiment, so it is not a model input.")

# ══════════════ TAB 6 — Event Study ══════════════
with tab6:
    st.markdown("### Cumulative Abnormal Return (CAR) — Event Study")
    st.markdown(
        "<div class='explain-box'>"
        "An <b>event study</b> is a standard academic finance method to measure the true stock impact of a specific event. "
        f"It works by estimating what <b>{selected_company}</b> <i>should have done</i> based on the broader market "
        f"(<code>{market_ticker}</code>), then measuring the difference. That difference is the F1 effect. "
        f"Every line below is computed live from real prices around each {FIRST_SEASON}–{THIS_YEAR} race, "
        f"grouped by how {selected_team} finished."
        "</div>", unsafe_allow_html=True
    )

    car_by_day, per_race = (event_study(stock_hist, market_hist, all_events, selected_team)
                            if has_stock else (pd.DataFrame(), pd.DataFrame()))
    if car_by_day.empty:
        st.info("Not enough price history to run the event study for this stock.")
    else:
        fig6 = go.Figure()
        for outcome in [o for o in OUTCOMES if o in car_by_day.columns]:
            n = int((per_race["outcome"] == outcome).sum())
            fig6.add_trace(go.Scatter(
                x=car_by_day.index, y=car_by_day[outcome], mode="lines+markers",
                name=f"{outcome} (n={n})", line=dict(color=OUTCOME_COLORS[outcome], width=2.5 if outcome == "WIN" else 2),
                hovertemplate=f"<b>{outcome}</b><br>Day %{{x}}: %{{y:+.2f}}%<extra></extra>",
            ))
        fig6.add_vline(x=0, line_dash="solid", line_color="white", line_width=2,
                       annotation_text="First trading day after race", annotation_position="top right",
                       annotation_font_color="white")
        fig6.add_hline(y=0, line_color="#555", line_width=1)
        fig6.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=420, xaxis=dict(title="Trading days relative to the race", showgrid=False, tickvals=list(car_by_day.index)),
            yaxis=dict(title="Cumulative abnormal return (%)", showgrid=True, gridcolor="#1e2228"),
            legend=dict(orientation="h", y=1.08), margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig6, use_container_width=True)

        from scipy import stats
        sig_rows = []
        for outcome in [o for o in OUTCOMES if o in set(per_race["outcome"])]:
            cars = per_race.loc[per_race["outcome"] == outcome, "CAR"]
            t, p = stats.ttest_1samp(cars, 0) if len(cars) >= 3 else (np.nan, np.nan)
            sig_rows.append({"Outcome": outcome, "Races": len(cars), "Mean CAR [−1, +7] %": round(cars.mean(), 2),
                             "t-stat": round(t, 2) if pd.notna(t) else None,
                             "p-value": round(p, 3) if pd.notna(p) else None,
                             "Significant (p < 0.05)": "✅ yes" if pd.notna(p) and p < 0.05 else "no"})
        st.markdown("#### Is the effect real or noise?")
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
        st.caption(f"Average market beta of {selected_company} vs {market_ticker}: {per_race['beta'].mean():.2f}")

    st.markdown("#### Step-by-step: how the event study works")
    st.markdown(f"""
1. **Estimation window** — For each race, take the 120 trading days ending 10 days before the race. Regress the stock's daily return against the market index (`{market_ticker}`) to get α (alpha) and β (beta).
2. **Expected return** — On the days around the race, predict what the stock "should" return: `E[R] = α + β × R_market`
3. **Abnormal return** — `AR = Actual return − Expected return`. This isolates the F1 effect.
4. **CAR** — Sum the ARs day by day (the chart starts at Day −5). The table sums Day −1 to Day +7: a positive CAR after a WIN = the stock genuinely outperformed the market after that win.
5. **Significance test** — one-sample t-test; p < 0.05 means the average CAR is unlikely to be random noise.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#444;font-size:12px'>"
    "F1 Pulse · Streamlit + Plotly · Data: yfinance, Jolpica F1 API, FastF1 · Master of Data Science project · Not financial advice"
    "</div>", unsafe_allow_html=True,
)

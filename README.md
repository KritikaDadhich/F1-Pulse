# 🏎️ F1 Pulse — Stock Impact Intelligence

> A live data science web app that analyses and predicts how Formula 1 race events affect the stock prices of publicly-traded companies linked to the sport.

**Live Demo:** [f1-pulse.streamlit.app](https://your-app.streamlit.app) *(replace with your URL after deployment)*

---

## What This Project Does

Formula 1 is uniquely connected to the global stock market. Ferrari (NYSE: RACE) is simultaneously a car manufacturer and an F1 team. Liberty Media owns F1 itself (NASDAQ: FWONA). When Ferrari wins at Monza, analysts notice. When Red Bull is penalised for a budget cap violation, their sponsors react.

**F1 Pulse tests one specific hypothesis:**
> *Can Formula 1 race outcomes — wins, crashes, championship changes — measurably predict short-term stock price movements in companies linked to the sport?*

The app answers this across three layers:

| Layer | Question | Method |
|-------|----------|--------|
| **Descriptive** | Does a relationship exist? | Price chart + event markers |
| **Statistical** | How significant is it? | Event study (CAR) + t-test |
| **Predictive** | Can we forecast it? | XGBoost + ARIMA ensemble |

---

## Live Features

- **Price & Events** — Live stock price chart with F1 race results overlaid as markers. Hover over any race marker to see the event details.
- **Teams & Sponsors** — All 11 F1 teams (2026 grid), their publicly-traded parent companies, title sponsors, and engine suppliers — mapped to verified stock tickers across 10 global exchanges.
- **F1 Race History** — Complete race results from 2021–2026, filterable by season and team, with win statistics and podium breakdowns.
- ** Driver Standings** — Championship points table and multi-season comparison chart for all 22 drivers.
- **Correlation Heatmap** — Real computed average stock returns by event type (WIN, PODIUM, DNF, CRASH) across Day +1 to Day +7 after each race.
- **ML Prediction** — Simulate any upcoming 2026 Grand Prix and predict whether the linked stock will move UP or DOWN, with confidence score.
- **Sentiment Analysis** — VADER NLP sentiment scores from real F1 news headlines via NewsAPI, plotted around race weekends.
- **Event Study (CAR)** — Academic-grade Cumulative Abnormal Return analysis with t-test statistical significance.

---

## Architecture

```
f1-pulse/
├── app.py                        # Main Streamlit dashboard (7 tabs)
├── train_model.py                # Offline model training script
├── requirements.txt
├── models/
│   ├── predictor.py              # XGBoost + ARIMA ensemble
│   └── xgb_model.pkl             # Pre-trained model (210 KB)
└── utils/
    ├── data_fetcher.py           # yfinance + static F1 event data
    ├── feature_engineering.py   # Event study + feature pipeline
    ├── teams_sponsors.py        # 11 teams, 37 companies, verified tickers
    ├── f1_data.py               # Race results + standings 2021–2026
    └── sentiment.py             # NewsAPI + VADER NLP
```

---

## Technical Methodology

### Data Pipeline
- **Stock prices** — `yfinance` (Yahoo Finance), daily OHLCV, primary exchange listings
- **F1 race results** — Static verified dataset covering 2021–2026 (150+ races)
- **News sentiment** — NewsAPI headlines, VADER sentiment scoring
- **Feature window** — Minimum 730 days (2 seasons) for statistically meaningful samples

### Feature Engineering
For each race event, the pipeline computes:
- `post_event_return` — actual % stock return over 3 trading days after the race
- `pre_event_volatility` — 10-day rolling return standard deviation before race
- `volume_spike` — race-day volume vs 20-day average (anomaly detector)
- `rsi_at_event` — Relative Strength Index on race day
- `rolling_win_rate` — team's win rate in last 5 races
- `event_score` — numeric encoding: WIN=3, PODIUM=2, DNF=-1, CRASH=-2

### Machine Learning
- **XGBoost classifier** — predicts stock direction (UP/DOWN) after upcoming races
- **Training data** — 981 historical F1-stock event pairs across 6 seasons and 11 companies
- **Cross-validation accuracy** — 68.2% (±1.0%) on held-out test set
- **ARIMA(2,1,2)** — time series baseline for 5-day price forecasting

### Event Study (Academic Finance Method)
1. **Estimation window** — 120 trading days before each race (days -130 to -10)
2. **Expected return** — stock's historical mean return from estimation window
3. **Abnormal return** — `AR = Actual return − Expected return`
4. **CAR** — cumulative sum of ARs from Day −5 to Day +7
5. **T-test** — two-sample t-test on WIN vs non-WIN returns (p < 0.05 threshold)

---

## Companies & Exchanges Tracked

| Category | Examples | Exchanges |
|----------|---------|-----------|
| Team manufacturers | Ferrari (RACE), Mercedes-Benz (MBG.DE), VW/Audi (VOW3.DE), GM/Cadillac (GM) | NYSE, XETRA |
| Title sponsors | Oracle (ORCL), HP (HPQ), Atlassian (TEAM), Visa (V), Google (GOOGL) | NYSE, NASDAQ |
| Engine partners | Honda (HMC), Ford (F) | NYSE |
| F1 ownership | Liberty Media (FWONA) | NASDAQ |
| Tyre supplier | Pirelli (PIRC.MI) | Borsa Italiana |
| Global sponsors | LVMH (MC.PA), PepsiCo (PEP), Salesforce (CRM), AWS (AMZN) | EURONEXT, NASDAQ |
| Betting partners | Flutter/FanDuel (FLUT) | NYSE |

**37 verified public companies · 10 global exchanges · 11 countries**

> All tickers use each company's **primary exchange listing** — not US OTC ADRs — for accurate, liquid pricing.

---

## Run Locally

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/KritikaDadhich/f1-pulse.git
cd f1-pulse

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train the ML model (creates models/xgb_model.pkl)
python train_model.py

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Optional: Add NewsAPI key for real sentiment data
1. Get a free key at [newsapi.org](https://newsapi.org) (no credit card)
2. Create `.streamlit/secrets.toml`:
```toml
NEWSAPI_KEY = "your_key_here"
```

---

## Academic Context

This project was built as part of a **Master of Data Science** at **RMIT University** (2025–2026). It directly applies coursework from:

| Subject | Application in this project |
|---------|----------------------------|
| Practical Data Science with Python | Full pipeline: `app.py`, all utils |
| Data Wrangling using R → Python | `feature_engineering.py` merge logic |
| Advanced Programming with Data Science | XGBoost pipeline, scikit-learn |
| Time Series & Forecasting using R | ARIMA model in `predictor.py` |
| Data Visualisation using R → Plotly | All 7 dashboard tabs |
| Applied Analytics | Event study, CAR methodology |
| Data Mining using Weka → XGBoost | Classification model training |
| Cloud Computing (AWS) | Architecture designed for Lambda/S3 |

---

## Future Roadmap

- [ ] AWS Lambda function for daily automated data refresh
- [ ] DynamoDB storage for prediction tracking and accuracy logging
- [ ] Multivariate regression across multiple stocks simultaneously
- [ ] Real-time race weekend alerts via email/SMS
- [ ] Expand to MotoGP and NASCAR (same methodology, different companies)
- [ ] Portfolio simulator: "what if you bought Ferrari stock before every win?"

---

## Disclaimer

This project is built for academic and educational purposes only. Nothing in this app constitutes financial advice. All stock data is sourced from public APIs with standard delays. Past F1-stock correlations do not guarantee future returns.

---

## Author

**Kritika Dadhich**
Master of Data Science · RMIT University · Melbourne, Australia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-kritikadadhich-blue?logo=linkedin)](https://www.linkedin.com/in/kritikadadhich)
[![GitHub](https://img.shields.io/badge/GitHub-KritikaDadhich-black?logo=github)](https://github.com/KritikaDadhich)

---

*Built with Streamlit · Plotly · XGBoost · yfinance · VADER NLP*

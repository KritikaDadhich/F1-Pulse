"""
sentiment.py
Fetches F1-related news headlines via NewsAPI and scores them using VADER.
VADER (Valence Aware Dictionary and sEntiment Reasoner) is ideal for short news text.
Free NewsAPI key: https://newsapi.org (100 requests/day on free tier)
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("[sentiment] vaderSentiment not installed. Run: pip install vaderSentiment")


NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # Set in .env or Streamlit secrets

F1_KEYWORDS = [
    "Formula 1", "F1 Grand Prix", "Ferrari F1",
    "Mercedes F1", "Red Bull Racing", "McLaren F1",
]


def get_sentiment_score(race_name: str, race_date: str, days_before: int = 3) -> dict:
    """
    Fetch headlines around a race weekend and return aggregated VADER sentiment.
    Returns: { "compound": float, "positive": float, "negative": float, "article_count": int }
    """
    if not NEWSAPI_KEY:
        return _mock_sentiment(race_name)

    end_dt   = pd.to_datetime(race_date) + timedelta(days=1)
    start_dt = end_dt - timedelta(days=days_before)

    headlines = []
    for keyword in [race_name, "Formula 1"]:
        articles = _fetch_headlines(keyword, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        headlines.extend(articles)

    if not headlines:
        return _mock_sentiment(race_name)

    return _score_headlines(headlines)


def get_sentiment_series(start: str, end: str, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a time series of sentiment scores aligned to race dates.
    """
    rows = []
    for _, ev in events_df.iterrows():
        score = get_sentiment_score(ev.get("race", ""), str(ev["date"]))
        rows.append({
            "date":          ev["date"],
            "race":          ev.get("race", ""),
            "compound":      score["compound"],
            "positive":      score["positive"],
            "negative":      score["negative"],
            "article_count": score["article_count"],
        })
    return pd.DataFrame(rows)


def _fetch_headlines(query: str, from_date: str, to_date: str) -> list:
    """Call NewsAPI /everything endpoint."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "to":       to_date,
        "language": "en",
        "sortBy":   "relevancy",
        "pageSize": 30,
        "apiKey":   NEWSAPI_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a.get("title", "") + " " + (a.get("description") or "") for a in articles]
    except Exception as e:
        print(f"[sentiment] NewsAPI error: {e}")
        return []


def _score_headlines(headlines: list) -> dict:
    """Run VADER on each headline, return averaged compound score."""
    if not VADER_AVAILABLE:
        return _mock_sentiment("unknown")

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h) for h in headlines if h.strip()]

    if not scores:
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "article_count": 0}

    return {
        "compound":      sum(s["compound"] for s in scores) / len(scores),
        "positive":      sum(s["pos"]      for s in scores) / len(scores),
        "negative":      sum(s["neg"]      for s in scores) / len(scores),
        "article_count": len(scores),
    }


def _mock_sentiment(race_name: str) -> dict:
    """Return plausible mock sentiment when API key not set."""
    import hashlib, math
    h = int(hashlib.md5(race_name.encode()).hexdigest(), 16)
    compound = (h % 200 - 100) / 200
    return {
        "compound":      round(compound, 3),
        "positive":      round(max(0, compound) * 0.8, 3),
        "negative":      round(max(0, -compound) * 0.8, 3),
        "article_count": (h % 20) + 5,
    }

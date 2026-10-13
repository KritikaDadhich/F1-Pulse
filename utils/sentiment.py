"""
sentiment.py
Fetches recent F1 news headlines via NewsAPI and scores them with VADER.
VADER (Valence Aware Dictionary and sEntiment Reasoner) is built for short news text.
Free NewsAPI key: https://newsapi.org — set it as NEWSAPI_KEY in Streamlit secrets or .env.
"""
import os
import requests
from datetime import datetime, timedelta

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("[sentiment] vaderSentiment not installed. Run: pip install vaderSentiment")


def get_news_sentiment(team: str, api_key: str = "", days: int = 7) -> dict:
    """
    Average VADER sentiment of the last `days` of English headlines about `team` in F1.
    Returns {"compound", "positive", "negative", "article_count", "headlines", "error"}.
    `compound` is None when no score could be computed — never a made-up value.
    """
    api_key = api_key or os.getenv("NEWSAPI_KEY", "")
    empty = {"compound": None, "positive": None, "negative": None, "article_count": 0, "headlines": []}
    if not api_key:
        return {**empty, "error": "No NEWSAPI_KEY configured"}
    if not VADER_AVAILABLE:
        return {**empty, "error": "vaderSentiment is not installed"}

    params = {
        "q":        f'"{team}" AND (F1 OR "Formula 1" OR "Grand Prix")',
        "searchIn": "title,description",   # match the team in the headline, not deep in the article body
        "from":     (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 50,
        "apiKey":   api_key,
    }
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            return {**empty, "error": data.get("message", f"NewsAPI HTTP {resp.status_code}")}
        articles = data.get("articles", [])
    except Exception as e:
        return {**empty, "error": f"NewsAPI unreachable: {e}"}

    analyzer = SentimentIntensityAnalyzer()
    scored = []
    for a in articles:
        text = f"{a.get('title') or ''} {a.get('description') or ''}".strip()
        if text:
            scored.append((analyzer.polarity_scores(text), a))
    if not scored:
        return {**empty, "error": f"No recent headlines found for {team}"}

    n = len(scored)
    return {
        "compound":      sum(s["compound"] for s, _ in scored) / n,
        "positive":      sum(s["pos"] for s, _ in scored) / n,
        "negative":      sum(s["neg"] for s, _ in scored) / n,
        "article_count": n,
        "headlines":     [{"title": a.get("title", ""), "source": (a.get("source") or {}).get("name", ""),
                           "url": a.get("url", ""), "published": (a.get("publishedAt") or "")[:10],
                           "score": round(s["compound"], 2)} for s, a in scored[:8]],
        "error":         None,
    }

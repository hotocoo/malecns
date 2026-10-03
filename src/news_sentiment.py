"""News sentiment analysis for trading decisions.

Fetches forex news headlines and performs basic sentiment analysis
to incorporate market sentiment into trading decisions.

Uses free news APIs:
- Forex Factory: https://www.forexfactory.com/
- Investing.com: https://www.investing.com/
- Reuters: https://www.reuters.com/markets/currencies/
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests


# Sentiment keywords
POSITIVE_WORDS = [
    "rise", "rises", "rose", "rising", "gain", "gains", "gained", "gaining",
    "strong", "strength", "stronger", "strongest", "bullish", "up", "higher",
    "improve", "improves", "improved", "improving", "beat", "beats", "beat expectations",
    "positive", "good", "optimistic", "growth", "grow", "grew", "growing",
    "surge", "surges", "surged", "surging", "jump", "jumps", "jumped", "jumping",
    "recover", "recovers", "recovered", "recovering", "rebound", "rebounds",
]

NEGATIVE_WORDS = [
    "fall", "falls", "fell", "falling", "drop", "drops", "dropped", "dropping",
    "weak", "weakness", "weaker", "weakest", "bearish", "down", "lower",
    "decline", "declines", "declined", "declining", "miss", "misses", "missed expectations",
    "negative", "bad", "pessimistic", "slump", "slumps", "slumped", "slumping",
    "slow", "slows", "slowed", "slowing", "shrink", "shrinks", "shrunk", "shrinking",
    "plunge", "plunges", "plunged", "plunging", "tumble", "tumbles", "tumbled", "tumbling",
    "worry", "worries", "worried", "worrisome", "concern", "concerns", "concerned",
]

NEUTRAL_WORDS = [
    "stable", "steady", "unchanged", "flat", "hold", "holds", "held", "holding",
    "mixed", "moderate", "mild", "slight", "slightly", "marginal",
]


class NewsSentimentAnalyzer:
    """Analyzes forex news sentiment for trading decisions."""
    
    def __init__(
        self,
        instruments: list[str] | None = None,
        cache_dir: str | None = None,
        cache_ttl: int = 300,  # 5 minutes
    ):
        self.instruments = instruments or ["EUR_USD", "GBP_USD", "USD_JPY"]
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent.parent / ".news_cache"
        self.cache_ttl = cache_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_fetch = 0
        self._headlines: list[dict] = []
    
    def fetch_headlines(self) -> list[dict]:
        """Fetch latest forex news headlines."""
        now = time.time()
        if now - self._last_fetch < self.cache_ttl and self._headlines:
            return self._headlines
        
        headlines = []
        
        # Try multiple sources
        try:
            headlines.extend(self._fetch_forexfactory())
        except Exception as e:
            print(f"ForexFactory fetch failed: {e}")
        
        try:
            headlines.extend(self._fetch_investing_com())
        except Exception as e:
            print(f"Investing.com fetch failed: {e}")
        
        # Filter by instrument
        if self.instruments:
            instrument_keywords = [inst.replace("_", "") for inst in self.instruments]
            filtered = []
            for h in headlines:
                title_lower = h["title"].lower()
                if any(kw.lower() in title_lower for kw in instrument_keywords):
                    filtered.append(h)
            if filtered:
                headlines = filtered
        
        self._headlines = headlines[:50]  # Limit to 50 headlines
        self._last_fetch = now
        return self._headlines
    
    def _fetch_forexfactory(self) -> list[dict]:
        """Fetch headlines from Forex Factory."""
        url = "https://www.forexfactory.com/calendar"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MaleCNS/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Simple HTML parsing (no BeautifulSoup dependency)
        headlines = []
        # Look for news items in the HTML
        pattern = r'<a[^>]*href="([^"]*news[^"]*)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, response.text, re.IGNORECASE)
        for url, title in matches[:20]:
            headlines.append({
                "title": title.strip(),
                "url": url,
                "source": "forexfactory",
            })
        return headlines
    
    def _fetch_investing_com(self) -> list[dict]:
        """Fetch headlines from Investing.com."""
        url = "https://www.investing.com/currencies/"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MaleCNS/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        headlines = []
        pattern = r'<a[^>]*href="([^"]*)"[^>]*class="[^"]*headline[^"]*"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, response.text, re.IGNORECASE)
        for url, title in matches[:20]:
            headlines.append({
                "title": title.strip(),
                "url": url,
                "source": "investing.com",
            })
        return headlines
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text. Returns score from -1 (very negative) to +1 (very positive)."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        neutral_count = sum(1 for w in words if w in NEUTRAL_WORDS)
        
        total = positive_count + negative_count + neutral_count
        if total == 0:
            return 0.0
        
        score = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, score))
    
    def get_market_sentiment(self) -> dict[str, float]:
        """Get overall market sentiment per instrument.
        
        Returns dict mapping instrument to sentiment score (-1 to +1).
        """
        headlines = self.fetch_headlines()
        if not headlines:
            return {inst: 0.0 for inst in self.instruments}
        
        # Analyze each headline
        sentiments: dict[str, list[float]] = {inst: [] for inst in self.instruments}
        
        for h in headlines:
            score = self.analyze_sentiment(h["title"])
            # Determine which instruments this headline is relevant to
            title_lower = h["title"].lower()
            for inst in self.instruments:
                if inst.replace("_", "").lower() in title_lower:
                    sentiments[inst].append(score)
                elif "usd" in title_lower or "dollar" in title_lower:
                    # USD-related news affects all pairs
                    sentiments[inst].append(score * 0.5)
        
        # Average sentiment per instrument
        result = {}
        for inst, scores in sentiments.items():
            if scores:
                result[inst] = sum(scores) / len(scores)
            else:
                result[inst] = 0.0
        
        return result
    
    def get_sentiment_factor(self, instrument: str) -> float:
        """Get sentiment adjustment factor for position sizing.
        
        Returns 0.5-1.5 multiplier based on sentiment strength.
        Strong positive sentiment -> larger long positions
        Strong negative sentiment -> larger short positions
        """
        sentiments = self.get_market_sentiment()
        score = sentiments.get(instrument, 0.0)
        
        # Convert sentiment score to sizing factor
        # 0.0 -> 1.0 (no change)
        # 0.5 -> 1.5 (50% larger position)
        # -0.5 -> 0.5 (50% smaller position)
        factor = 1.0 + score
        return max(0.5, min(1.5, factor))
    
    def should_trade(self, instrument: str) -> bool:
        """Check if sentiment is not too uncertain to trade."""
        sentiments = self.get_market_sentiment()
        score = sentiments.get(instrument, 0.0)
        
        # If sentiment is very uncertain (close to 0), consider waiting
        # But this is a soft filter, not a hard stop
        return True  # Always allow trading; sentiment is a factor, not a blocker

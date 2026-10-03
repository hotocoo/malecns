"""Multi-timeframe analysis for trading decisions.

Analyzes price data across multiple timeframes (1m, 5m, 15m, 1h, 4h, daily)
to identify trends and confluence for higher-confidence trading signals.

Higher timeframes determine trend direction, lower timeframes provide entry timing.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class MultiTimeframeAnalyzer:
    """Analyzes price action across multiple timeframes."""
    
    # Timeframe definitions (in minutes)
    TIMEFRAMES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "daily": 1440,
    }
    
    def __init__(
        self,
        timeframes: list[str] | None = None,
        ma_periods: list[int] | None = None,
    ):
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h"]
        self.ma_periods = ma_periods or [20, 50]
        self.price_history: dict[str, list[float]] = {}
        self._current_prices: dict[str, float] = {}
        
        # Initialize price history for each timeframe
        for tf in self.timeframes:
            self.price_history[tf] = []
    
    def update_price(self, instrument: str, price: float):
        """Update price for an instrument."""
        self._current_prices[instrument] = price
        
        # Add to 1m timeframe
        self.price_history["1m"].append(price)
        
        # Aggregate to higher timeframes
        self._aggregate_timeframes()
    
    def _aggregate_timeframes(self):
        """Aggregate 1m prices to higher timeframes using OHLC."""
        for tf in self.timeframes:
            if tf == "1m":
                continue
            
            tf_minutes = self.TIMEFRAMES[tf]
            one_min_prices = self.price_history["1m"]
            
            if len(one_min_prices) < tf_minutes:
                continue
            
            # Use most recent tf_minutes of 1m data
            recent = one_min_prices[-tf_minutes:]
            
            # Simple aggregation: use the last price
            self.price_history[tf].append(recent[-1])
    
    def get_ma(self, timeframe: str, period: int) -> float | None:
        """Get moving average for a timeframe and period."""
        prices = self.price_history.get(timeframe, [])
        if len(prices) < period:
            return None
        
        return float(np.mean(prices[-period:]))
    
    def get_trend(self, timeframe: str) -> float:
        """Get trend direction for a timeframe.
        
        Returns:
            +1.0: Strong uptrend
            +0.5: Weak uptrend
            0.0: No trend / sideways
            -0.5: Weak downtrend
            -1.0: Strong downtrend
        """
        prices = self.price_history.get(timeframe, [])
        if len(prices) < 5:
            return 0.0
        
        # Calculate trend using linear regression slope
        n = len(prices)
        x = np.arange(n)
        y = np.array(prices)
        
        # Normalize y to percentage changes
        if y[0] == 0:
            return 0.0
        y_pct = (y - y[0]) / y[0]
        
        # Linear regression
        slope = np.polyfit(x, y_pct, 1)[0]
        
        # Normalize slope to -1 to +1 range
        # A 1% move over the period is considered strong
        trend = np.clip(slope * 100, -1.0, 1.0)
        return float(trend)
    
    def get_support_resistance(self, timeframe: str) -> tuple[float, float]:
        """Get support and resistance levels for a timeframe.
        
        Returns (support, resistance) as price levels.
        """
        prices = self.price_history.get(timeframe, [])
        if len(prices) < 10:
            return (0.0, 0.0)
        
        # Use recent 50 prices
        recent = prices[-50:] if len(prices) >= 50 else prices
        
        support = float(np.min(recent))
        resistance = float(np.max(recent))
        
        return (support, resistance)
    
    def get_confluence_score(self, instrument: str) -> float:
        """Calculate trend confluence across timeframes.
        
        High confluence = all timeframes agree on direction.
        Low confluence = timeframes disagree (choppy market).
        
        Returns score from -1.0 (all bearish) to +1.0 (all bullish).
        """
        if not self.timeframes:
            return 0.0
        
        trends = []
        weights = []
        
        # Higher timeframes get more weight
        tf_order = ["1m", "5m", "15m", "1h", "4h", "daily"]
        
        for tf in self.timeframes:
            trend = self.get_trend(tf)
            trends.append(trend)
            
            # Weight: higher timeframes = more important
            if tf in tf_order:
                weight = tf_order.index(tf) + 1
            else:
                weight = 1
            weights.append(weight)
        
        if not trends:
            return 0.0
        
        # Weighted average trend
        trends_arr = np.array(trends)
        weights_arr = np.array(weights)
        
        confluence = float(np.average(trends_arr, weights=weights_arr))
        return confluence
    
    def get_entry_signal(self, instrument: str) -> str:
        """Get entry signal based on multi-timeframe analysis.
        
        Returns:
            "buy": All timeframes bullish
            "sell": All timeframes bearish
            "buy_weak": Mostly bullish
            "sell_weak": Mostly bearish
            "hold": Mixed signals
        """
        confluence = self.get_confluence_score(instrument)
        
        if confluence > 0.5:
            return "buy"
        elif confluence > 0.2:
            return "buy_weak"
        elif confluence < -0.5:
            return "sell"
        elif confluence < -0.2:
            return "sell_weak"
        else:
            return "hold"
    
    def should_trade(self, instrument: str, threshold: float = 0.3) -> bool:
        """Check if multi-timeframe analysis suggests trading.
        
        Trades when there's sufficient trend confluence.
        """
        confluence = abs(self.get_confluence_score(instrument))
        return confluence > threshold
    
    def get_position_adjustment(self, instrument: str) -> float:
        """Get position size adjustment based on trend strength.
        
        Returns 0.5-1.5 multiplier:
        - Strong trend confluence -> larger position
        - Weak/mixed signals -> smaller position
        """
        confluence = abs(self.get_confluence_score(instrument))
        
        # Scale from 0.5 to 1.5 based on confluence strength
        adjustment = 0.5 + confluence
        return float(np.clip(adjustment, 0.5, 1.5))

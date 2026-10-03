"""Market regime detection for adaptive trading.

Detects whether the market is in a trending or ranging regime and
adjusts trading strategy accordingly:
- Trending: use trend-following strategies, wider stops
- Ranging: use mean-reversion strategies, tighter stops

Uses multiple indicators to determine market regime:
- ADX (Average Directional Index)
- Bollinger Band width
- Price vs moving average distance
"""

from __future__ import annotations

import numpy as np
from typing import Any


class MarketRegimeDetector:
    """Detects market regime (trending vs ranging)."""
    
    def __init__(
        self,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        trend_threshold: float = 25.0,  # ADX above this = trending
        bb_width_threshold: float = 0.02,  # BB width above this = trending
    ):
        self.adx_period = adx_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.trend_threshold = trend_threshold
        self.bb_width_threshold = bb_width_threshold
        
        self.price_history: dict[str, list[float]] = {}
        self.regime: dict[str, str] = {}  # instrument -> "trending" or "ranging"
        self.trend_direction: dict[str, str] = {}  # instrument -> "up" or "down"
    
    def update_price(self, instrument: str, price: float):
        """Update price history and regime detection."""
        if instrument not in self.price_history:
            self.price_history[instrument] = []
        
        self.price_history[instrument].append(price)
        
        # Keep only necessary history
        max_history = max(self.adx_period, self.bb_period) * 2
        if len(self.price_history[instrument]) > max_history:
            self.price_history[instrument].pop(0)
        
        # Detect regime
        self._detect_regime(instrument)
    
    def _detect_regime(self, instrument: str):
        """Detect market regime using multiple indicators."""
        prices = self.price_history[instrument]
        n = len(prices)
        
        if n < self.bb_period:
            self.regime[instrument] = "unknown"
            self.trend_direction[instrument] = "unknown"
            return
        
        # Calculate ADX
        adx = self._calculate_adx(prices)
        
        # Calculate Bollinger Band width
        bb_width = self._calculate_bb_width(prices)
        
        # Determine regime
        is_trending = False
        
        # ADX check
        if adx > self.trend_threshold:
            is_trending = True
        
        # BB width check
        if bb_width > self.bb_width_threshold:
            is_trending = True
        
        if is_trending:
            self.regime[instrument] = "trending"
            # Determine trend direction
            ma = np.mean(prices[-self.bb_period:])
            if prices[-1] > ma:
                self.trend_direction[instrument] = "up"
            else:
                self.trend_direction[instrument] = "down"
        else:
            self.regime[instrument] = "ranging"
            self.trend_direction[instrument] = "unknown"
    
    def _calculate_adx(self, prices: list[float]) -> float:
        """Calculate Average Directional Index (ADX)."""
        n = len(prices)
        if n < self.adx_period + 1:
            return 0.0
        
        # Calculate directional movement
        plus_dm = 0.0
        minus_dm = 0.0
        tr_sum = 0.0
        
        for i in range(n - self.adx_period, n):
            high = max(prices[i], prices[i-1])
            low = min(prices[i], prices[i-1])
            
            # True Range
            tr = high - low
            tr_sum += tr
            
            # Directional Movement
            up_move = prices[i] - prices[i-1]
            down_move = prices[i-1] - prices[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm += up_move
            elif down_move > up_move and down_move > 0:
                minus_dm += down_move
        
        if tr_sum == 0:
            return 0.0
        
        # Normalize
        plus_di = (plus_dm / tr_sum) * 100
        minus_di = (minus_dm / tr_sum) * 100
        
        # ADX
        if plus_di + minus_di == 0:
            return 0.0
        
        adx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        return float(adx)
    
    def _calculate_bb_width(self, prices: list[float]) -> float:
        """Calculate Bollinger Band width."""
        n = len(prices)
        if n < self.bb_period:
            return 0.0
        
        window = prices[-self.bb_period:]
        mean = np.mean(window)
        std = np.std(window)
        
        if mean == 0:
            return 0.0
        
        bb_width = (2 * self.bb_std * std) / mean
        return float(bb_width)
    
    def get_regime(self, instrument: str) -> str:
        """Get current market regime for instrument."""
        return self.regime.get(instrument, "unknown")
    
    def get_trend_direction(self, instrument: str) -> str:
        """Get current trend direction for instrument."""
        return self.trend_direction.get(instrument, "unknown")
    
    def should_trade(self, instrument: str) -> bool:
        """Check if conditions are favorable for trading."""
        regime = self.get_regime(instrument)
        return regime in ["trending", "ranging"]
    
    def get_position_adjustment(self, instrument: str) -> float:
        """Get position size adjustment based on regime.
        
        Returns:
            0.5 - 1.5 multiplier for position size
        """
        regime = self.get_regime(instrument)
        
        if regime == "trending":
            # Larger positions in trending markets
            return 1.2
        elif regime == "ranging":
            # Smaller positions in ranging markets
            return 0.8
        else:
            return 1.0
    
    def get_stop_loss_adjustment(self, instrument: str) -> float:
        """Get stop loss adjustment based on regime.
        
        Returns:
            0.5 - 2.0 multiplier for stop loss percentage
        """
        regime = self.get_regime(instrument)
        
        if regime == "trending":
            # Wider stops in trending markets
            return 1.5
        elif regime == "ranging":
            # Tighter stops in ranging markets
            return 0.7
        else:
            return 1.0

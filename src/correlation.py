"""Currency pair correlation analysis for trading decisions.

Analyzes correlations between currency pairs to identify:
- Pairs that move together (positive correlation)
- Pairs that move in opposite directions (negative correlation)
- Divergence opportunities when correlations break down

Uses historical price data to calculate rolling correlation coefficients.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any


class CorrelationAnalyzer:
    """Analyzes correlations between currency pairs."""
    
    def __init__(
        self,
        instruments: list[str],
        window: int = 50,  # Rolling window size
    ):
        self.instruments = instruments
        self.window = window
        self.price_history: dict[str, list[float]] = {inst: [] for inst in instruments}
        self.max_history = window
    
    def update_prices(self, prices: dict[str, float]):
        """Update price history with new prices."""
        for inst, price in prices.items():
            if inst in self.price_history:
                self.price_history[inst].append(price)
                if len(self.price_history[inst]) > self.max_history:
                    self.price_history[inst].pop(0)
    
    def get_returns(self, instrument: str) -> np.ndarray:
        """Calculate returns for an instrument."""
        prices = self.price_history[instrument]
        if len(prices) < 2:
            return np.array([])
        
        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        return returns
    
    def calculate_correlation(self, inst1: str, inst2: str) -> float:
        """Calculate correlation between two instruments."""
        returns1 = self.get_returns(inst1)
        returns2 = self.get_returns(inst2)
        
        # Need at least 5 data points for meaningful correlation
        min_len = min(len(returns1), len(returns2))
        if min_len < 5:
            return 0.0
        
        # Use most recent data
        returns1 = returns1[-min_len:]
        returns2 = returns2[-min_len:]
        
        # Calculate Pearson correlation
        try:
            corr = np.corrcoef(returns1, returns2)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except Exception:
            return 0.0
    
    def get_correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Calculate correlation matrix for all instruments."""
        matrix = {}
        for inst1 in self.instruments:
            matrix[inst1] = {}
            for inst2 in self.instruments:
                matrix[inst1][inst2] = self.calculate_correlation(inst1, inst2)
        return matrix
    
    def get_divergence_score(self, instrument: str) -> float:
        """Calculate divergence score for an instrument.
        
        High divergence means the instrument is moving differently from its
        correlated pairs, which may indicate a trading opportunity.
        """
        if len(self.instruments) < 2:
            return 0.0
        
        # Find most correlated pair
        max_corr = 0.0
        correlated_inst = None
        for other in self.instruments:
            if other != instrument:
                corr = abs(self.calculate_correlation(instrument, other))
                if corr > max_corr:
                    max_corr = corr
                    correlated_inst = other
        
        if correlated_inst is None or max_corr < 0.5:
            return 0.0
        
        # Calculate divergence: difference in recent returns
        returns1 = self.get_returns(instrument)
        returns2 = self.get_returns(correlated_inst)
        
        if len(returns1) < 5 or len(returns2) < 5:
            return 0.0
        
        # Recent 5-period returns
        recent1 = np.mean(returns1[-5:])
        recent2 = np.mean(returns2[-5:])
        
        # Divergence = difference in returns / expected correlation
        divergence = abs(recent1 - recent2) / (1.0 - max_corr + 0.001)
        return float(divergence)
    
    def should_trade(self, instrument: str, threshold: float = 0.005) -> bool:
        """Check if correlation analysis suggests trading this instrument.
        
        Trades when:
        1. Correlation is stable (not diverging)
        2. Or divergence is high (opportunity to fade)
        """
        divergence = self.get_divergence_score(instrument)
        
        # If divergence is very high, the correlation may have broken
        # This is a signal to be cautious
        if divergence > threshold * 3:
            return False
        
        return True
    
    def get_position_adjustment(self, instrument: str) -> float:
        """Get position size adjustment based on correlation.
        
        Returns 0.5-1.5 multiplier:
        - High correlation with other positions -> smaller position (redundancy)
        - Low correlation -> larger position (diversification benefit)
        """
        if len(self.instruments) < 2:
            return 1.0
        
        # Average correlation with other instruments
        correlations = []
        for other in self.instruments:
            if other != instrument:
                corr = abs(self.calculate_correlation(instrument, other))
                correlations.append(corr)
        
        if not correlations:
            return 1.0
        
        avg_corr = np.mean(correlations)
        
        # High correlation -> smaller position
        # Low correlation -> larger position
        adjustment = 1.0 + (0.5 - avg_corr)
        return float(np.clip(adjustment, 0.5, 1.5))

"""Risk parity position sizing for portfolio optimization.

Allocates risk equally across positions so that each position contributes
equally to portfolio risk, regardless of position size or volatility.

This prevents any single position from dominating portfolio risk.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class RiskParitySizer:
    """Calculates position sizes using risk parity methodology."""
    
    def __init__(
        self,
        instruments: list[str],
        target_risk_per_position: float = 0.01,  # 1% of portfolio per position
        max_positions: int = 5,
        volatility_window: int = 20,
    ):
        self.instruments = instruments
        self.target_risk_per_position = target_risk_per_position
        self.max_positions = max_positions
        self.volatility_window = volatility_window
        self.price_history: dict[str, list[float]] = {inst: [] for inst in instruments}
        self.positions: dict[str, float] = {}  # instrument -> units
    
    def update_prices(self, prices: dict[str, float]):
        """Update price history."""
        for inst, price in prices.items():
            if inst in self.price_history:
                self.price_history[inst].append(price)
                if len(self.price_history[inst]) > self.volatility_window:
                    self.price_history[inst].pop(0)
    
    def get_volatility(self, instrument: str) -> float:
        """Calculate realized volatility for an instrument."""
        prices = self.price_history[instrument]
        if len(prices) < 2:
            return 0.02  # Default 2%
        
        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        volatility = float(np.std(returns))
        
        # Ensure minimum volatility
        return max(volatility, 0.001)
    
    def calculate_position_size(
        self,
        instrument: str,
        price: float,
        portfolio_value: float,
        stop_loss_pct: float = 0.02,
    ) -> float:
        """Calculate position size using risk parity.
        
        Each position should contribute equally to portfolio risk.
        
        Args:
            instrument: Currency pair symbol
            price: Current price
            portfolio_value: Total portfolio value
            stop_loss_pct: Stop loss percentage
        
        Returns:
            Position size in units
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0
        
        # Target risk per position
        target_risk = portfolio_value * self.target_risk_per_position
        
        # Risk per unit = price * stop_loss_pct
        risk_per_unit = price * stop_loss_pct
        
        # Position size = target risk / risk per unit
        position_size = target_risk / risk_per_unit
        
        # Adjust for volatility (higher vol -> smaller position)
        volatility = self.get_volatility(instrument)
        position_size *= (0.02 / volatility)  # Normalize to 2% volatility
        
        return float(position_size)
    
    def get_portfolio_risk_allocation(self) -> dict[str, float]:
        """Calculate risk allocation across all positions.
        
        Returns dict mapping instrument to risk percentage.
        """
        total_risk = 0.0
        risk_by_instrument = {}
        
        for inst, units in self.positions.items():
            if inst not in self.price_history or not self.price_history[inst]:
                continue
            
            price = self.price_history[inst][-1]
            volatility = self.get_volatility(inst)
            
            # Risk = units * price * volatility
            risk = units * price * volatility
            risk_by_instrument[inst] = risk
            total_risk += risk
        
        # Convert to percentages
        if total_risk > 0:
            for inst in risk_by_instrument:
                risk_by_instrument[inst] = risk_by_instrument[inst] / total_risk
        
        return risk_by_instrument
    
    def get_rebalancing_suggestions(self, portfolio_value: float) -> list[dict]:
        """Get suggestions for rebalancing the portfolio.
        
        Returns list of {instrument, action, units, reason} dicts.
        """
        suggestions = []
        risk_alloc = self.get_portfolio_risk_allocation()
        
        for inst in self.instruments:
            if inst not in self.positions:
                continue
            
            current_risk = risk_alloc.get(inst, 0.0)
            target_risk = 1.0 / len(self.positions) if self.positions else 0.0
            
            # If risk allocation is off by more than 20%, suggest rebalancing
            if abs(current_risk - target_risk) > target_risk * 0.2:
                if current_risk > target_risk:
                    action = "reduce"
                    reason = f"Risk too high ({current_risk:.1%} vs {target_risk:.1%} target)"
                else:
                    action = "increase"
                    reason = f"Risk too low ({current_risk:.1%} vs {target_risk:.1%} target)"
                
                suggestions.append({
                    "instrument": inst,
                    "action": action,
                    "reason": reason,
                })
        
        return suggestions
    
    def update_position(self, instrument: str, units: float):
        """Update tracked position."""
        self.positions[instrument] = units
    
    def remove_position(self, instrument: str):
        """Remove tracked position."""
        if instrument in self.positions:
            del self.positions[instrument]

"""Drawdown protection for trading portfolio.

Monitors portfolio drawdown and adjusts trading parameters dynamically:
- Reduces position sizes during drawdown
- Tightens stop losses
- Reduces maximum number of positions
- Can trigger full pause of trading if drawdown exceeds threshold

This protects capital during losing streaks and prevents overtrading.
"""

from __future__ import annotations

import time
from typing import Any


class DrawdownProtection:
    """Monitors and protects against portfolio drawdown."""
    
    def __init__(
        self,
        initial_balance: float,
        max_drawdown_pct: float = 0.10,  # Stop trading at 10% drawdown
        warning_drawdown_pct: float = 0.05,  # Reduce risk at 5% drawdown
        recovery_factor: float = 0.5,  # Recover 50% of drawdown to resume
    ):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown_pct = max_drawdown_pct
        self.warning_drawdown_pct = warning_drawdown_pct
        self.recovery_factor = recovery_factor
        
        self.current_balance = initial_balance
        self.drawdown = 0.0
        self.drawdown_pct = 0.0
        self.is_paused = False
        self.risk_multiplier = 1.0
        self.last_update = time.time()
    
    def update_balance(self, balance: float):
        """Update current balance and calculate drawdown."""
        self.current_balance = balance
        self.last_update = time.time()
        
        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Calculate drawdown
        self.drawdown = self.peak_balance - balance
        self.drawdown_pct = self.drawdown / self.peak_balance if self.peak_balance > 0 else 0.0
        
        # Adjust risk multiplier based on drawdown
        self._adjust_risk()
    
    def _adjust_risk(self):
        """Adjust risk parameters based on drawdown level."""
        if self.drawdown_pct >= self.max_drawdown_pct:
            # Maximum drawdown: pause trading
            self.is_paused = True
            self.risk_multiplier = 0.0
        elif self.drawdown_pct >= self.warning_drawdown_pct:
            # Warning drawdown: reduce risk
            self.is_paused = False
            # Linear reduction from 1.0 to 0.5 as drawdown goes from warning to max
            reduction = (self.drawdown_pct - self.warning_drawdown_pct) / (self.max_drawdown_pct - self.warning_drawdown_pct)
            self.risk_multiplier = 1.0 - (reduction * 0.5)
        else:
            # Normal: full risk
            self.is_paused = False
            self.risk_multiplier = 1.0
    
    def get_risk_multiplier(self) -> float:
        """Get current risk multiplier (0.0 to 1.0)."""
        return self.risk_multiplier
    
    def should_trade(self) -> bool:
        """Check if trading should continue."""
        return not self.is_paused
    
    def get_adjusted_position_size(self, base_size: float) -> float:
        """Get position size adjusted for drawdown."""
        return base_size * self.risk_multiplier
    
    def get_adjusted_max_positions(self, base_max: int) -> int:
        """Get max positions adjusted for drawdown."""
        if self.is_paused:
            return 0
        # Reduce max positions by half during warning drawdown
        if self.drawdown_pct >= self.warning_drawdown_pct:
            return max(1, base_max // 2)
        return base_max
    
    def get_adjusted_stop_loss(self, base_sl_pct: float) -> float:
        """Get stop loss percentage adjusted for drawdown.
        
        Tighter stop losses during drawdown to limit further losses.
        """
        if self.is_paused:
            return base_sl_pct
        # Tighten stop loss by 50% during warning drawdown
        if self.drawdown_pct >= self.warning_drawdown_pct:
            return base_sl_pct * 0.5
        return base_sl_pct
    
    def get_status(self) -> dict:
        """Get current drawdown protection status."""
        return {
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "drawdown": self.drawdown,
            "drawdown_pct": self.drawdown_pct,
            "risk_multiplier": self.risk_multiplier,
            "is_paused": self.is_paused,
            "max_drawdown_pct": self.max_drawdown_pct,
            "warning_drawdown_pct": self.warning_drawdown_pct,
        }

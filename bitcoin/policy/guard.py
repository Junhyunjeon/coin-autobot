"""Guard rules for risk management."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import timedelta


class GuardRules:
    """Apply guard rules for trade filtering and risk management."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize guard rules.
        
        Args:
            config: Guard configuration
        """
        self.cooldown_minutes = config.get('cooldown_minutes_after_loss', 30)
        self.max_trades_per_day = config.get('max_trades_per_day', 20)
        self.max_spread_bps = config.get('max_spread_bps', 10)
        
        # Track trading history
        self.trade_history = []
        self.last_loss_time = None
    
    def check_cooldown(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if we're in cooldown period.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if trading is allowed (not in cooldown)
        """
        if self.last_loss_time is None:
            return True
        
        cooldown_end = self.last_loss_time + timedelta(minutes=self.cooldown_minutes)
        return timestamp >= cooldown_end
    
    def check_daily_limit(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if daily trade limit is reached.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if trading is allowed (under limit)
        """
        if not self.trade_history:
            return True
        
        # Count trades today
        today_start = timestamp.normalize()
        today_trades = [
            t for t in self.trade_history 
            if t['timestamp'] >= today_start
        ]
        
        return len(today_trades) < self.max_trades_per_day
    
    def check_spread(self, spread_bps: float) -> bool:
        """
        Check if spread is acceptable.
        
        Args:
            spread_bps: Current spread in basis points
            
        Returns:
            True if trading is allowed (spread acceptable)
        """
        return spread_bps <= self.max_spread_bps
    
    def check_volatility_limit(self, 
                              current_vol: float,
                              threshold_vol: float = 0.05) -> bool:
        """
        Check if volatility is within acceptable range.
        
        Args:
            current_vol: Current volatility
            threshold_vol: Maximum acceptable volatility
            
        Returns:
            True if trading is allowed
        """
        return current_vol <= threshold_vol
    
    def check_drawdown_limit(self, 
                            portfolio_value: float,
                            peak_value: float,
                            max_drawdown: float = 0.20) -> bool:
        """
        Check if drawdown is within limit.
        
        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            max_drawdown: Maximum allowed drawdown
            
        Returns:
            True if trading is allowed
        """
        if peak_value <= 0:
            return True
        
        current_drawdown = (peak_value - portfolio_value) / peak_value
        return current_drawdown < max_drawdown
    
    def apply_guards(self,
                    signal: int,
                    timestamp: pd.Timestamp,
                    spread_bps: float = 0,
                    current_vol: float = 0,
                    portfolio_value: float = None,
                    peak_value: float = None) -> int:
        """
        Apply all guard rules to a signal.
        
        Args:
            signal: Trading signal
            timestamp: Current timestamp
            spread_bps: Current spread
            current_vol: Current volatility
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            
        Returns:
            Filtered signal (0 if blocked by guards)
        """
        if signal == 0:
            return 0
        
        # Check all guard conditions
        guards_passed = [
            self.check_cooldown(timestamp),
            self.check_daily_limit(timestamp),
            self.check_spread(spread_bps),
            self.check_volatility_limit(current_vol) if current_vol > 0 else True,
        ]
        
        # Check drawdown if portfolio values provided
        if portfolio_value is not None and peak_value is not None:
            guards_passed.append(
                self.check_drawdown_limit(portfolio_value, peak_value)
            )
        
        # If any guard fails, block the signal
        if not all(guards_passed):
            return 0
        
        return signal
    
    def record_trade(self, 
                    timestamp: pd.Timestamp,
                    signal: int,
                    pnl: float = 0):
        """
        Record a trade for tracking.
        
        Args:
            timestamp: Trade timestamp
            signal: Trade signal
            pnl: Trade P&L (if closed)
        """
        self.trade_history.append({
            'timestamp': timestamp,
            'signal': signal,
            'pnl': pnl
        })
        
        # Update last loss time if this was a loss
        if pnl < 0:
            self.last_loss_time = timestamp
    
    def get_guard_stats(self, 
                        signals_checked: int,
                        signals_blocked: int) -> Dict[str, Any]:
        """Get guard statistics."""
        return {
            'total_signals_checked': signals_checked,
            'signals_blocked_by_guards': signals_blocked,
            'guard_block_rate': signals_blocked / signals_checked if signals_checked > 0 else 0,
            'total_trades_recorded': len(self.trade_history),
            'cooldown_active': self.last_loss_time is not None,
            'max_trades_per_day': self.max_trades_per_day,
            'max_spread_bps': self.max_spread_bps
        }
    
    def reset(self):
        """Reset guard state."""
        self.trade_history = []
        self.last_loss_time = None
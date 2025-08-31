"""Volatility targeting position sizing."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class VolatilityTargeting:
    """Position sizing based on volatility targeting."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize volatility targeting.
        
        Args:
            config: Sizing configuration
        """
        self.target_annual_vol = config.get('target_annual_vol', 0.20)
        self.min_pos = config.get('min_pos', 0.0)
        self.max_pos = config.get('max_pos', 1.0)
        
        # Convert annual to minute volatility (assuming 365 * 24 * 60 minutes)
        self.minutes_per_year = 365 * 24 * 60
        self.target_minute_vol = self.target_annual_vol / np.sqrt(self.minutes_per_year)
    
    def calculate_position_size(self,
                               current_vol: float,
                               signal_strength: float = 1.0,
                               current_position: float = 0.0) -> float:
        """
        Calculate position size based on volatility.
        
        Args:
            current_vol: Current market volatility (per minute)
            signal_strength: Signal confidence (0 to 1)
            current_position: Current position size
            
        Returns:
            Target position size
        """
        if current_vol <= 0:
            return 0.0
        
        # Base position size from volatility targeting
        vol_ratio = self.target_minute_vol / current_vol
        base_size = min(vol_ratio, 2.0)  # Cap leverage at 2x
        
        # Adjust by signal strength
        target_size = base_size * signal_strength
        
        # Apply position limits
        target_size = np.clip(target_size, self.min_pos, self.max_pos)
        
        return target_size
    
    def calculate_dynamic_sizing(self,
                                df: pd.DataFrame,
                                signals: pd.Series,
                                vol_window: int = 60) -> pd.Series:
        """
        Calculate dynamic position sizes for all signals.
        
        Args:
            df: OHLCV data
            signals: Trading signals
            vol_window: Window for volatility calculation
            
        Returns:
            Position sizes
        """
        # Calculate rolling volatility
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(vol_window).std()
        
        # Initialize position sizes
        position_sizes = pd.Series(0.0, index=signals.index)
        
        for idx in signals.index:
            if signals.loc[idx] != 0:
                # Get current volatility
                current_vol = rolling_vol.loc[idx]
                
                # Calculate signal strength (can be enhanced with confidence scores)
                signal_strength = np.abs(signals.loc[idx])
                
                # Calculate position size
                size = self.calculate_position_size(
                    current_vol,
                    signal_strength
                )
                
                # Apply sign based on signal direction
                position_sizes.loc[idx] = size * np.sign(signals.loc[idx])
        
        return position_sizes
    
    def apply_risk_limits(self,
                         position_sizes: pd.Series,
                         df: pd.DataFrame) -> pd.Series:
        """
        Apply additional risk limits to position sizes.
        
        Args:
            position_sizes: Calculated position sizes
            df: OHLCV data
            
        Returns:
            Risk-adjusted position sizes
        """
        adjusted_sizes = position_sizes.copy()
        
        # Calculate drawdown
        cum_returns = df['close'].pct_change().cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns - running_max
        
        # Reduce position size during drawdowns
        for idx in position_sizes.index:
            if idx in drawdown.index:
                dd = drawdown.loc[idx]
                if dd < -0.05:  # 5% drawdown
                    scale = max(0.5, 1 + dd * 2)  # Reduce size up to 50%
                    adjusted_sizes.loc[idx] *= scale
        
        return adjusted_sizes
    
    def calculate_kelly_sizing(self,
                             win_rate: float,
                             avg_win: float,
                             avg_loss: float) -> float:
        """
        Calculate Kelly criterion position size.
        
        Args:
            win_rate: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
            
        Returns:
            Kelly fraction
        """
        if avg_loss <= 0:
            return 0.0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / avg_loss
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b
        
        # Apply Kelly fraction scaling (typically use 25% of full Kelly)
        kelly_fraction *= 0.25
        
        return max(0, min(kelly_fraction, self.max_pos))
    
    def get_position_metrics(self, position_sizes: pd.Series) -> Dict[str, float]:
        """Calculate position sizing metrics."""
        non_zero = position_sizes[position_sizes != 0]
        
        if len(non_zero) == 0:
            return {
                'avg_position_size': 0,
                'max_position_size': 0,
                'min_position_size': 0,
                'position_utilization': 0
            }
        
        return {
            'avg_position_size': float(np.abs(non_zero).mean()),
            'max_position_size': float(np.abs(position_sizes).max()),
            'min_position_size': float(np.abs(non_zero).min()),
            'position_utilization': float(len(non_zero) / len(position_sizes)),
            'long_ratio': float((position_sizes > 0).sum() / len(non_zero)),
            'short_ratio': float((position_sizes < 0).sum() / len(non_zero))
        }
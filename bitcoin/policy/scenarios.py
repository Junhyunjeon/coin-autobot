"""Scenario-based trading policy engine."""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


class ScenarioPolicy:
    """Parse and apply scenario-based trading policies."""
    
    def __init__(self, scenarios_path: str):
        """
        Initialize scenario policy.
        
        Args:
            scenarios_path: Path to scenarios.yaml file
        """
        self.scenarios_path = scenarios_path
        self.scenarios = self._load_scenarios()
        self.active_scenario = None
    
    def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load scenarios from YAML file."""
        if not Path(self.scenarios_path).exists():
            return []
        
        with open(self.scenarios_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data or 'scenarios' not in data:
            return []
        
        return data['scenarios']
    
    def detect_market_regime(self, df: pd.DataFrame, lookback: int = 100) -> str:
        """
        Detect current market regime.
        
        Args:
            df: OHLCV data
            lookback: Lookback period for regime detection
            
        Returns:
            Regime name: 'trend', 'range', 'volatile', 'quiet'
        """
        if len(df) < lookback:
            return 'unknown'
        
        recent_data = df.iloc[-lookback:]
        returns = recent_data['close'].pct_change()
        
        # Calculate regime indicators
        volatility = returns.std()
        trend_strength = abs(returns.mean()) / volatility if volatility > 0 else 0
        
        # ADX for trend detection (if available)
        if 'adx' in recent_data.columns:
            adx_mean = recent_data['adx'].mean()
            is_trending = adx_mean > 25
        else:
            # Simple trend detection using price movement
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            is_trending = abs(price_change) > 0.05
        
        # Classify regime
        if volatility > returns.rolling(lookback).std().mean() * 1.5:
            return 'volatile'
        elif volatility < returns.rolling(lookback).std().mean() * 0.5:
            return 'quiet'
        elif is_trending:
            return 'trend'
        else:
            return 'range'
    
    def calculate_spread(self, orderbook_df: Optional[pd.DataFrame], 
                        ohlcv_df: pd.DataFrame,
                        timestamp: pd.Timestamp) -> float:
        """
        Calculate bid-ask spread in basis points.
        
        Args:
            orderbook_df: Orderbook data (optional)
            ohlcv_df: OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Spread in basis points
        """
        if orderbook_df is not None and timestamp in orderbook_df.index:
            row = orderbook_df.loc[timestamp]
            if 'bid_px' in row and 'ask_px' in row:
                mid_price = (row['bid_px'] + row['ask_px']) / 2
                spread = (row['ask_px'] - row['bid_px']) / mid_price * 10000
                return spread
        
        # Fallback to high-low spread proxy
        if timestamp in ohlcv_df.index:
            row = ohlcv_df.loc[timestamp]
            spread = (row['high'] - row['low']) / row['close'] * 10000
            return spread
        
        return 0
    
    def select_scenario(self, 
                       df: pd.DataFrame,
                       orderbook_df: Optional[pd.DataFrame],
                       timestamp: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """
        Select applicable scenario based on current conditions.
        
        Args:
            df: OHLCV data
            orderbook_df: Orderbook data (optional)
            timestamp: Current timestamp
            
        Returns:
            Selected scenario or None
        """
        if not self.scenarios:
            return None
        
        # Detect current regime
        regime = self.detect_market_regime(df)
        
        # Calculate current spread
        spread_bps = self.calculate_spread(orderbook_df, df, timestamp)
        
        # Find matching scenario
        for scenario in self.scenarios:
            filters = scenario.get('filters', {})
            
            # Check regime filter
            if 'regime' in filters:
                if filters['regime'] != regime:
                    continue
            
            # Check spread filter
            if 'max_spread_bps' in filters:
                if spread_bps > filters['max_spread_bps']:
                    continue
            
            # All filters passed
            return scenario
        
        return None
    
    def apply_policy(self,
                    signal: int,
                    probability: float,
                    scenario: Optional[Dict[str, Any]]) -> int:
        """
        Apply scenario policy to signal.
        
        Args:
            signal: Original signal (-1, 0, 1)
            probability: Signal probability
            scenario: Active scenario
            
        Returns:
            Filtered signal
        """
        if scenario is None:
            return signal
        
        actions = scenario.get('actions', {})
        
        # Check allowed actions
        if 'allow' in actions:
            signal_action = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[signal]
            if signal_action not in actions['allow']:
                return 0  # PASS
        
        # Check disallowed actions
        if 'disallow' in actions:
            signal_action = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[signal]
            if signal_action in actions['disallow']:
                return 0  # PASS
        
        # Check pass conditions
        if 'pass_when' in actions:
            pass_when = actions['pass_when']
            if 'prob_between' in pass_when:
                lower, upper = pass_when['prob_between']
                if lower <= probability <= upper:
                    return 0  # PASS
        
        return signal
    
    def filter_signals(self,
                      signals: pd.Series,
                      probabilities: pd.Series,
                      df: pd.DataFrame,
                      orderbook_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Filter all signals based on scenarios.
        
        Args:
            signals: Original signals
            probabilities: Signal probabilities
            df: OHLCV data
            orderbook_df: Orderbook data (optional)
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        
        for timestamp in signals.index:
            if timestamp not in df.index:
                continue
            
            # Select scenario for this timestamp
            scenario = self.select_scenario(df.loc[:timestamp], orderbook_df, timestamp)
            
            # Apply policy
            filtered_signal = self.apply_policy(
                signals.loc[timestamp],
                probabilities.loc[timestamp] if timestamp in probabilities.index else 0.5,
                scenario
            )
            
            filtered_signals.loc[timestamp] = filtered_signal
        
        return filtered_signals
    
    def get_policy_stats(self,
                        original_signals: pd.Series,
                        filtered_signals: pd.Series) -> Dict[str, Any]:
        """Calculate policy filtering statistics."""
        total_signals = (original_signals != 0).sum()
        filtered_out = ((original_signals != 0) & (filtered_signals == 0)).sum()
        
        return {
            'total_original_signals': int(total_signals),
            'signals_filtered_by_policy': int(filtered_out),
            'policy_filter_rate': float(filtered_out / total_signals) if total_signals > 0 else 0,
            'scenarios_loaded': len(self.scenarios),
            'buy_signals_remaining': int((filtered_signals == 1).sum()),
            'sell_signals_remaining': int((filtered_signals == -1).sum())
        }
"""Feature engineering for trading signals."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class FeatureEngineer:
    """Generate technical features for model training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature configuration
        """
        self.returns_lags = config.get('returns_lags', [1, 5, 15, 60])
        self.vol_lookback = config.get('vol_lookback', 60)
        self.momentum_lookback = config.get('momentum_lookback', [20, 60])
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the dataset."""
        features_df = df.copy()
        
        # Price returns
        features_df = self._add_return_features(features_df)
        
        # Volatility features
        features_df = self._add_volatility_features(features_df)
        
        # Momentum indicators
        features_df = self._add_momentum_features(features_df)
        
        # Volume features
        features_df = self._add_volume_features(features_df)
        
        # Technical indicators
        features_df = self._add_technical_indicators(features_df)
        
        # Microstructure features
        features_df = self._add_microstructure_features(features_df)
        
        return features_df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        for lag in self.returns_lags:
            df[f'return_{lag}'] = df['close'].pct_change(lag)
            df[f'log_return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        
        # Cumulative returns
        for window in [5, 15, 30]:
            df[f'cum_return_{window}'] = df['close'].pct_change().rolling(window).sum()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        returns = df['close'].pct_change()
        
        # Historical volatility
        for window in [20, 60, 120]:
            df[f'volatility_{window}'] = returns.rolling(window).std()
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            np.log(df['high'] / df['low']).rolling(self.vol_lookback).apply(
                lambda x: np.sum(x**2) / (4 * len(x) * np.log(2))
            )
        )
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low'])**2 - 
            (2*np.log(2) - 1) * np.log(df['close'] / df['open'])**2
        ).rolling(self.vol_lookback).mean()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [14, 28]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        for period in [14, 28]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        # ADX (Average Directional Index)
        df['adx'] = self._calculate_adx(df, 14)
        
        # Rate of change
        for period in self.momentum_lookback:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        for window in [5, 20, 60]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume-price trend
        df['vpt'] = df['volume'] * ((df['close'] - df['close'].shift()) / df['close'].shift())
        df['vpt_cumsum'] = df['vpt'].cumsum()
        
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        mfi_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional technical indicators."""
        # Bollinger Bands
        for window in [20, 40]:
            ma = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'bb_upper_{window}'] = ma + 2 * std
            df[f'bb_lower_{window}'] = ma - 2 * std
            df[f'bb_width_{window}'] = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / df[f'bb_width_{window}']
        
        # Support/Resistance levels
        for window in [20, 60]:
            df[f'resistance_{window}'] = df['high'].rolling(window).max()
            df[f'support_{window}'] = df['low'].rolling(window).min()
            df[f'sr_position_{window}'] = (df['close'] - df[f'support_{window}']) / \
                                          (df[f'resistance_{window}'] - df[f'support_{window}'])
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Spread proxy (if no orderbook data)
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Roll's spread estimator
        returns = df['close'].pct_change()
        df['roll_spread'] = 2 * np.sqrt(-returns.rolling(20).cov(returns.shift()))
        
        # Kyle's lambda (price impact)
        df['kyle_lambda'] = returns.rolling(20).apply(
            lambda x: np.abs(x).sum() / df['volume'].rolling(20).sum().iloc[-1] if df['volume'].rolling(20).sum().iloc[-1] > 0 else 0
        )
        
        # Amihud illiquidity
        df['amihud'] = np.abs(returns) / df['volume']
        df['amihud_ma'] = df['amihud'].rolling(20).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.concat([
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift()),
            np.abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        feature_patterns = [
            'return_', 'log_return_', 'cum_return_',
            'volatility_', 'parkinson_vol', 'gk_vol', 'atr',
            'rsi_', 'macd', 'stoch_', 'adx', 'roc_',
            'volume_ma_', 'volume_ratio_', 'obv', 'vpt', 'mfi',
            'bb_', 'resistance_', 'support_', 'sr_position_',
            'hl_spread', 'roll_spread', 'kyle_lambda', 'amihud'
        ]
        return feature_patterns
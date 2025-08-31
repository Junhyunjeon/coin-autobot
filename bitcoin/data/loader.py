"""Data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoader:
    """Load and preprocess OHLCV and orderbook data."""
    
    def __init__(self, tz: str = "UTC"):
        self.tz = tz
    
    def load_ohlcv(self, path: str) -> pd.DataFrame:
        """Load OHLCV data from CSV."""
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df.index = df.index.tz_localize(self.tz, ambiguous='infer')
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def load_orderbook(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load L2 orderbook data if available."""
        if not path or not Path(path).exists():
            return None
        
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df.index = df.index.tz_localize(self.tz, ambiguous='infer')
        
        return df
    
    def resample_ohlcv(self, df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
        """Resample OHLCV data to specified frequency."""
        return df.resample(freq).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    def calculate_returns(self, df: pd.DataFrame, periods: list = [1, 5, 15, 60]) -> pd.DataFrame:
        """Calculate returns for multiple periods."""
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        return df
    
    def calculate_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        return df
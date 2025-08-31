"""Event detection using CUSUM algorithm."""

import numpy as np
import pandas as pd
from typing import List, Tuple


class CUSUMEventDetector:
    """Detect significant events using CUSUM algorithm."""
    
    def __init__(self, threshold: float = 0.005):
        """
        Initialize CUSUM detector.
        
        Args:
            threshold: Cumulative sum threshold for event detection
        """
        self.threshold = threshold
    
    def detect_events(self, returns: pd.Series) -> pd.DatetimeIndex:
        """
        Detect events using CUSUM on returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            DatetimeIndex of event timestamps
        """
        returns = returns.dropna()
        
        # Standardize returns
        mean_ret = returns.mean()
        std_ret = returns.std()
        standardized = (returns - mean_ret) / std_ret if std_ret > 0 else returns
        
        # CUSUM calculation
        cusum_pos = np.zeros(len(standardized))
        cusum_neg = np.zeros(len(standardized))
        
        for i in range(1, len(standardized)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized.iloc[i])
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized.iloc[i])
        
        # Detect events when CUSUM exceeds threshold
        threshold_scaled = self.threshold / std_ret if std_ret > 0 else self.threshold
        events_pos = np.where(cusum_pos > threshold_scaled)[0]
        events_neg = np.where(cusum_neg < -threshold_scaled)[0]
        
        # Combine and get unique event indices
        event_indices = np.unique(np.concatenate([events_pos, events_neg]))
        
        # Filter to reduce event clustering
        filtered_events = self._filter_events(event_indices, min_gap=10)
        
        return returns.index[filtered_events]
    
    def _filter_events(self, events: np.ndarray, min_gap: int = 10) -> np.ndarray:
        """Filter events to maintain minimum gap between consecutive events."""
        if len(events) == 0:
            return events
        
        filtered = [events[0]]
        for event in events[1:]:
            if event - filtered[-1] >= min_gap:
                filtered.append(event)
        
        return np.array(filtered)
    
    def calculate_event_statistics(self, df: pd.DataFrame, events: pd.DatetimeIndex) -> pd.DataFrame:
        """Calculate statistics for detected events."""
        stats = []
        
        for event_time in events:
            idx = df.index.get_loc(event_time)
            
            # Look ahead window for event impact
            window = 20
            if idx + window < len(df):
                future_returns = df['close'].iloc[idx:idx+window].pct_change().cumsum()
                max_return = future_returns.max()
                min_return = future_returns.min()
                
                stats.append({
                    'timestamp': event_time,
                    'price': df['close'].iloc[idx],
                    'volume': df['volume'].iloc[idx],
                    'max_future_return': max_return,
                    'min_future_return': min_return,
                    'volatility': df['close'].iloc[max(0,idx-20):idx].pct_change().std()
                })
        
        return pd.DataFrame(stats) if stats else pd.DataFrame()
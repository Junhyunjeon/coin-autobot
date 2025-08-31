"""Tests for data loader module."""

import pytest
import pandas as pd
import numpy as np
from bitcoin.data import DataLoader, CUSUMEventDetector, TripleBarrierLabeler


class TestDataLoader:
    
    def test_load_ohlcv(self, tmp_path):
        """Test OHLCV data loading."""
        # Create sample CSV
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.random(100) * 40000,
            'high': np.random.random(100) * 40000 + 40000,
            'low': np.random.random(100) * 40000 + 39000,
            'close': np.random.random(100) * 40000 + 39500,
            'volume': np.random.random(100) * 1000
        })
        
        csv_path = tmp_path / "test_ohlcv.csv"
        sample_data.to_csv(csv_path, index=False)
        
        loader = DataLoader(tz='UTC')
        df = loader.load_ohlcv(str(csv_path))
        
        assert len(df) == 100
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert df.index.tz is not None


class TestCUSUMEventDetector:
    
    def test_detect_events(self):
        """Test CUSUM event detection."""
        # Create returns with obvious events
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.001, 1000))
        # Insert large moves
        returns.iloc[100] = 0.02
        returns.iloc[500] = -0.02
        
        returns.index = pd.date_range('2023-01-01', periods=1000, freq='1min')
        
        detector = CUSUMEventDetector(threshold=0.005)
        events = detector.detect_events(returns)
        
        assert len(events) > 0
        assert isinstance(events, pd.DatetimeIndex)


class TestTripleBarrierLabeler:
    
    def test_apply_triple_barrier(self):
        """Test triple barrier labeling."""
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=1000, freq='1min', tz='UTC')
        df = pd.DataFrame({
            'open': 40000 + np.random.normal(0, 100, 1000).cumsum(),
            'high': 40000 + np.random.normal(0, 100, 1000).cumsum() + 50,
            'low': 40000 + np.random.normal(0, 100, 1000).cumsum() - 50,
            'close': 40000 + np.random.normal(0, 100, 1000).cumsum(),
            'volume': np.random.random(1000) * 1000
        }, index=dates)
        
        # Create some events
        events = dates[::100]  # Every 100 periods
        
        labeler = TripleBarrierLabeler(pt_mult=2.0, sl_mult=1.0, max_holding_minutes=60)
        labels_df = labeler.apply_triple_barrier(df, events)
        
        assert len(labels_df) > 0
        assert 'label' in labels_df.columns
        assert all(label in [-1, 0, 1] for label in labels_df['label'])
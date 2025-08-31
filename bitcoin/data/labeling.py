"""Triple barrier labeling for supervised learning."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TripleBarrierLabeler:
    """Generate labels using triple barrier method."""
    
    def __init__(self, 
                 pt_mult: float = 2.0,
                 sl_mult: float = 1.0, 
                 max_holding_minutes: int = 240):
        """
        Initialize triple barrier labeler.
        
        Args:
            pt_mult: Profit-take multiplier for volatility
            sl_mult: Stop-loss multiplier for volatility
            max_holding_minutes: Maximum holding period in minutes
        """
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.max_holding_minutes = max_holding_minutes
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 60) -> pd.Series:
        """Calculate rolling volatility."""
        returns = df['close'].pct_change()
        return returns.rolling(window).std()
    
    def apply_triple_barrier(self, 
                            df: pd.DataFrame,
                            events: pd.DatetimeIndex,
                            volatility: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply triple barrier labeling to events.
        
        Returns:
            DataFrame with columns: [event_time, label, return, holding_time, exit_type]
            label: 1 (profit), -1 (loss), 0 (timeout)
        """
        if volatility is None:
            volatility = self.calculate_volatility(df)
        
        labels = []
        
        for event_time in events:
            try:
                idx = df.index.get_loc(event_time)
                if idx >= len(df) - 1:
                    continue
                
                # Get volatility at event time
                vol = volatility.iloc[idx]
                if pd.isna(vol) or vol <= 0:
                    continue
                
                # Set barriers
                entry_price = df['close'].iloc[idx]
                pt_barrier = entry_price * (1 + self.pt_mult * vol)
                sl_barrier = entry_price * (1 - self.sl_mult * vol)
                
                # Find exit point
                max_idx = min(idx + self.max_holding_minutes, len(df))
                future_prices = df['close'].iloc[idx+1:max_idx]
                
                if len(future_prices) == 0:
                    continue
                
                # Check which barrier is hit first
                pt_hit = future_prices >= pt_barrier
                sl_hit = future_prices <= sl_barrier
                
                label = 0  # Default to timeout
                exit_idx = max_idx - 1
                exit_type = 'timeout'
                
                if pt_hit.any() or sl_hit.any():
                    if pt_hit.any() and sl_hit.any():
                        pt_idx = pt_hit.idxmax()
                        sl_idx = sl_hit.idxmax()
                        if df.index.get_loc(pt_idx) < df.index.get_loc(sl_idx):
                            label = 1
                            exit_idx = df.index.get_loc(pt_idx)
                            exit_type = 'profit_take'
                        else:
                            label = -1
                            exit_idx = df.index.get_loc(sl_idx)
                            exit_type = 'stop_loss'
                    elif pt_hit.any():
                        label = 1
                        exit_idx = df.index.get_loc(pt_hit.idxmax())
                        exit_type = 'profit_take'
                    else:
                        label = -1
                        exit_idx = df.index.get_loc(sl_hit.idxmax())
                        exit_type = 'stop_loss'
                
                # Calculate actual return
                exit_price = df['close'].iloc[exit_idx]
                actual_return = (exit_price - entry_price) / entry_price
                holding_time = exit_idx - idx
                
                labels.append({
                    'event_time': event_time,
                    'label': label,
                    'return': actual_return,
                    'holding_time': holding_time,
                    'exit_type': exit_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'volatility': vol,
                    'pt_barrier': pt_barrier,
                    'sl_barrier': sl_barrier
                })
                
            except Exception as e:
                print(f"Error processing event at {event_time}: {e}")
                continue
        
        return pd.DataFrame(labels)
    
    def generate_metalabels(self, 
                           df: pd.DataFrame,
                           primary_signals: pd.Series,
                           events: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate meta-labels for secondary model.
        
        Args:
            df: OHLCV data
            primary_signals: Primary model predictions (1: buy, -1: sell, 0: hold)
            events: Event timestamps
            
        Returns:
            DataFrame with meta-labels (1: take signal, 0: ignore signal)
        """
        # Apply triple barrier to get ground truth
        labels_df = self.apply_triple_barrier(df, events)
        
        metalabels = []
        
        for _, row in labels_df.iterrows():
            event_time = row['event_time']
            
            # Get primary signal at event time
            if event_time in primary_signals.index:
                primary_signal = primary_signals.loc[event_time]
                
                # Meta-label is 1 if primary signal direction matches actual outcome
                if primary_signal != 0:
                    if (primary_signal > 0 and row['label'] == 1) or \
                       (primary_signal < 0 and row['label'] == -1):
                        metalabel = 1
                    else:
                        metalabel = 0
                    
                    metalabels.append({
                        'event_time': event_time,
                        'primary_signal': primary_signal,
                        'actual_label': row['label'],
                        'metalabel': metalabel,
                        'return': row['return'],
                        'holding_time': row['holding_time']
                    })
        
        return pd.DataFrame(metalabels)
    
    def calculate_label_statistics(self, labels_df: pd.DataFrame) -> dict:
        """Calculate statistics for labeled data."""
        if labels_df.empty:
            return {}
        
        stats = {
            'total_events': len(labels_df),
            'profit_ratio': (labels_df['label'] == 1).mean(),
            'loss_ratio': (labels_df['label'] == -1).mean(),
            'timeout_ratio': (labels_df['label'] == 0).mean(),
            'avg_return': labels_df['return'].mean(),
            'avg_holding_time': labels_df['holding_time'].mean(),
            'sharpe_ratio': labels_df['return'].mean() / labels_df['return'].std() if labels_df['return'].std() > 0 else 0
        }
        
        return stats
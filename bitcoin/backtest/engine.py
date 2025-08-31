"""Event-driven backtest engine."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .costs import CostModel


@dataclass
class Position:
    """Track position state."""
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    unrealized_pnl: float = 0.0
    
    
@dataclass
class Trade:
    """Record individual trades."""
    timestamp: pd.Timestamp
    side: str  # 'buy' or 'sell'
    size: float
    price: float
    execution_price: float
    fee: float
    slippage_cost: float
    signal: int
    

class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.initial_capital = config.get('initial_capital', 100000)
        self.cost_model = CostModel(config.get('costs', {}))
        
        # Portfolio state
        self.cash = self.initial_capital
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.peak_equity = self.initial_capital
        
        # Performance tracking
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'total_slippage': 0
        }
    
    def reset(self):
        """Reset backtest state."""
        self.cash = self.initial_capital
        self.position = Position()
        self.trades = []
        self.equity_curve = []
        self.peak_equity = self.initial_capital
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'total_slippage': 0
        }
    
    def execute_signal(self,
                      timestamp: pd.Timestamp,
                      signal: int,
                      size: float,
                      price: float,
                      volume: float = None,
                      orderbook_row: Optional[pd.Series] = None) -> Optional[Trade]:
        """
        Execute a trading signal.
        
        Args:
            timestamp: Signal timestamp
            signal: Trading signal (-1, 0, 1)
            size: Position size
            price: Current price
            volume: Recent volume
            orderbook_row: Orderbook snapshot
            
        Returns:
            Trade object if executed
        """
        if signal == 0 or size == 0:
            return None
        
        # Determine trade side
        side = 'buy' if signal > 0 else 'sell'
        
        # Calculate costs
        costs = self.cost_model.calculate_total_cost(
            price, size, side, volume, orderbook_row
        )
        
        # Check if we have enough capital
        trade_value = abs(size * costs['execution_price'])
        total_cost = trade_value + costs['total_cost']
        
        if self.cash < total_cost:
            return None  # Not enough capital
        
        # Execute trade
        trade = Trade(
            timestamp=timestamp,
            side=side,
            size=size,
            price=price,
            execution_price=costs['execution_price'],
            fee=costs['fee'],
            slippage_cost=costs['slippage_cost'],
            signal=signal
        )
        
        # Update position
        if signal > 0:  # Buy
            self.position.size += size
            self.position.entry_price = (
                (self.position.entry_price * (self.position.size - size) + 
                 costs['execution_price'] * size) / self.position.size
            )
            self.cash -= total_cost
        else:  # Sell
            self.position.size -= size
            self.cash += trade_value - costs['total_cost']
        
        if abs(self.position.size) < 1e-10:
            self.position.size = 0
            self.position.entry_price = 0
        
        self.position.entry_time = timestamp
        
        # Record trade
        self.trades.append(trade)
        self.stats['total_trades'] += 1
        self.stats['total_fees'] += costs['fee']
        self.stats['total_slippage'] += costs['slippage_cost']
        
        return trade
    
    def update_position(self, current_price: float):
        """Update position P&L."""
        if self.position.size != 0:
            self.position.unrealized_pnl = (
                (current_price - self.position.entry_price) * self.position.size
            )
    
    def close_position(self,
                      timestamp: pd.Timestamp,
                      price: float,
                      volume: float = None,
                      orderbook_row: Optional[pd.Series] = None) -> Optional[Trade]:
        """
        Close current position.
        
        Args:
            timestamp: Close timestamp
            price: Current price
            volume: Recent volume
            orderbook_row: Orderbook snapshot
            
        Returns:
            Trade object if closed
        """
        if self.position.size == 0:
            return None
        
        # Execute closing trade
        signal = -1 if self.position.size > 0 else 1
        trade = self.execute_signal(
            timestamp, signal, abs(self.position.size),
            price, volume, orderbook_row
        )
        
        if trade:
            # Calculate realized P&L
            if self.position.size > 0:  # Was long
                pnl = (trade.execution_price - self.position.entry_price) * abs(self.position.size)
            else:  # Was short
                pnl = (self.position.entry_price - trade.execution_price) * abs(self.position.size)
            
            pnl -= (trade.fee + trade.slippage_cost)
            
            # Update stats
            self.stats['total_pnl'] += pnl
            if pnl > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
        
        return trade
    
    def get_equity(self, current_price: float) -> float:
        """Calculate current equity."""
        self.update_position(current_price)
        return self.cash + self.position.unrealized_pnl
    
    def run_backtest(self,
                    df: pd.DataFrame,
                    signals: pd.Series,
                    position_sizes: pd.Series,
                    orderbook_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run backtest on signals.
        
        Args:
            df: OHLCV data
            signals: Trading signals
            position_sizes: Position sizes for each signal
            orderbook_df: Orderbook data (optional)
            
        Returns:
            Backtest results
        """
        self.reset()
        
        # Align data
        common_index = df.index.intersection(signals.index)
        
        for timestamp in common_index:
            price = df.loc[timestamp, 'close']
            volume = df.loc[timestamp, 'volume']
            signal = signals.loc[timestamp]
            size = abs(position_sizes.loc[timestamp]) if timestamp in position_sizes.index else 0
            
            # Get orderbook if available
            orderbook_row = None
            if orderbook_df is not None and timestamp in orderbook_df.index:
                orderbook_row = orderbook_df.loc[timestamp]
            
            # Handle signal
            if signal != 0 and size > 0:
                # Close opposite position first
                if (signal > 0 and self.position.size < 0) or \
                   (signal < 0 and self.position.size > 0):
                    self.close_position(timestamp, price, volume, orderbook_row)
                
                # Open new position
                if self.position.size == 0:
                    self.execute_signal(timestamp, signal, size, price, volume, orderbook_row)
            
            # Update equity
            equity = self.get_equity(price)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.cash,
                'position_value': self.position.size * price if self.position.size != 0 else 0,
                'position_size': self.position.size,
                'unrealized_pnl': self.position.unrealized_pnl
            })
            
            # Update peak equity
            self.peak_equity = max(self.peak_equity, equity)
        
        # Close final position
        if self.position.size != 0:
            final_price = df.iloc[-1]['close']
            self.close_position(df.index[-1], final_price)
        
        # Calculate final metrics
        results = self.calculate_metrics()
        results['trades'] = self._format_trades()
        results['equity_curve'] = pd.DataFrame(self.equity_curve)
        
        return results
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        returns = equity_df['equity'].pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate
        win_rate = self.stats['winning_trades'] / self.stats['total_trades'] if self.stats['total_trades'] > 0 else 0
        
        # Average trade
        avg_trade = self.stats['total_pnl'] / self.stats['total_trades'] if self.stats['total_trades'] > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.stats['total_trades'],
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'avg_trade_pnl': avg_trade,
            'total_pnl': self.stats['total_pnl'],
            'total_fees': self.stats['total_fees'],
            'total_slippage': self.stats['total_slippage'],
            'final_equity': equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital
        }
    
    def _format_trades(self) -> pd.DataFrame:
        """Format trades for output."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'side': trade.side,
                'size': trade.size,
                'price': trade.price,
                'execution_price': trade.execution_price,
                'slippage_bps': (trade.execution_price - trade.price) / trade.price * 10000,
                'fee': trade.fee,
                'total_cost': trade.fee + trade.slippage_cost,
                'signal': trade.signal
            })
        
        return pd.DataFrame(trades_data)
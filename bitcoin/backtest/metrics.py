"""Performance metrics calculation."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class MetricsCalculator:
    """Calculate comprehensive performance metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              periods_per_year: int = 252*24*60) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Returns series
            periods_per_year: Number of periods in a year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                               periods_per_year: int = 252*24*60) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Returns series
            periods_per_year: Number of periods in a year
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        return mean_return / downside_std * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            (max_drawdown, peak_date, trough_date)
        """
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        max_dd = drawdown.min()
        
        if max_dd >= 0:
            return 0.0, None, None
        
        # Find peak and trough dates
        trough_idx = drawdown.idxmin()
        peak_idx = equity_curve[:trough_idx].idxmax()
        
        return max_dd, peak_idx, trough_idx
    
    @staticmethod
    def calculate_calmar_ratio(total_return: float,
                              max_drawdown: float,
                              years: float = 1.0) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            total_return: Total return
            max_drawdown: Maximum drawdown (negative value)
            years: Number of years
            
        Returns:
            Calmar ratio
        """
        if max_drawdown >= 0 or years <= 0:
            return 0.0
        
        annual_return = (1 + total_return) ** (1/years) - 1
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_win_loss_ratio(trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate win/loss statistics.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            Win/loss statistics
        """
        if trades_df.empty:
            return {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0
            }
        
        # Calculate P&L for each trade
        if 'pnl' not in trades_df.columns:
            # Estimate P&L from execution prices
            trades_df['pnl'] = 0  # Placeholder
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Expectancy
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    @staticmethod
    def calculate_turnover(trades_df: pd.DataFrame,
                          total_capital: float,
                          period_days: float) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            trades_df: DataFrame with trade data
            total_capital: Total capital
            period_days: Period in days
            
        Returns:
            Annual turnover rate
        """
        if trades_df.empty or total_capital <= 0 or period_days <= 0:
            return 0.0
        
        total_volume = (trades_df['size'] * trades_df['execution_price']).abs().sum()
        daily_turnover = total_volume / total_capital / period_days
        annual_turnover = daily_turnover * 365
        
        return annual_turnover
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  periods_per_year: int = 252*24*60) -> float:
        """
        Calculate Information ratio.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (if None, uses 0)
            periods_per_year: Number of periods in a year
            
        Returns:
            Information ratio
        """
        if benchmark_returns is None:
            active_returns = returns
        else:
            active_returns = returns - benchmark_returns
        
        if len(active_returns) < 2:
            return 0.0
        
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return active_returns.mean() / tracking_error * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_all_metrics(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            backtest_results: Backtest results dictionary
            
        Returns:
            Comprehensive metrics dictionary
        """
        equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        trades_df = backtest_results.get('trades', pd.DataFrame())
        
        if equity_curve.empty:
            return {}
        
        # Calculate returns
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Get basic metrics from backtest
        metrics = {
            'total_return': backtest_results.get('total_return', 0),
            'total_trades': backtest_results.get('total_trades', 0),
            'win_rate': backtest_results.get('win_rate', 0),
            'avg_trade_pnl': backtest_results.get('avg_trade_pnl', 0),
            'total_fees': backtest_results.get('total_fees', 0),
            'total_slippage': backtest_results.get('total_slippage', 0),
        }
        
        # Calculate additional metrics
        metrics['sharpe_ratio'] = MetricsCalculator.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = MetricsCalculator.calculate_sortino_ratio(returns)
        
        max_dd, peak_date, trough_date = MetricsCalculator.calculate_max_drawdown(equity_curve['equity'])
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_peak'] = peak_date
        metrics['max_drawdown_trough'] = trough_date
        
        # Calculate Calmar ratio
        period_days = (equity_curve['timestamp'].iloc[-1] - equity_curve['timestamp'].iloc[0]).days
        years = period_days / 365
        metrics['calmar_ratio'] = MetricsCalculator.calculate_calmar_ratio(
            metrics['total_return'], max_dd, years
        )
        
        # Win/loss statistics
        if not trades_df.empty:
            win_loss_stats = MetricsCalculator.calculate_win_loss_ratio(trades_df)
            metrics.update(win_loss_stats)
            
            # Turnover
            initial_capital = equity_curve['equity'].iloc[0]
            metrics['annual_turnover'] = MetricsCalculator.calculate_turnover(
                trades_df, initial_capital, period_days
            )
        
        # CAGR
        if years > 0:
            metrics['cagr'] = (1 + metrics['total_return']) ** (1/years) - 1
        else:
            metrics['cagr'] = 0
        
        # Risk-adjusted metrics
        metrics['return_over_max_dd'] = metrics['total_return'] / abs(max_dd) if max_dd < 0 else 0
        
        return metrics
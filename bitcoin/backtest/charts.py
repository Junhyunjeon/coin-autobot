"""Charting and visualization utilities."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


class ChartGenerator:
    """Generate backtest charts and visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize chart generator."""
        plt.style.use(style)
        sns.set_palette("husl")
        self.figsize = (15, 10)
    
    def plot_signal_overlay(self,
                           df: pd.DataFrame,
                           signals: pd.Series,
                           trades_df: pd.DataFrame = None,
                           save_path: str = None) -> plt.Figure:
        """
        Plot price with signal overlay.
        
        Args:
            df: OHLCV data
            signals: Trading signals
            trades_df: Executed trades
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, 
                                            gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1, color='black')
        
        # Overlay signals
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        if not buy_signals.empty:
            buy_prices = df.loc[buy_signals.index, 'close']
            ax1.scatter(buy_signals.index, buy_prices, 
                       color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
        
        if not sell_signals.empty:
            sell_prices = df.loc[sell_signals.index, 'close']
            ax1.scatter(sell_signals.index, sell_prices,
                       color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)
        
        # Mark executed trades if available
        if trades_df is not None and not trades_df.empty:
            for _, trade in trades_df.iterrows():
                color = 'darkgreen' if trade['side'] == 'buy' else 'darkred'
                ax1.axvline(x=trade['timestamp'], color=color, alpha=0.3, linestyle='--')
        
        ax1.set_title('Price and Trading Signals', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot volume
        ax2.bar(df.index, df['volume'], alpha=0.5, color='gray')
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot position indicator
        position_indicator = pd.Series(0, index=df.index)
        current_pos = 0
        for timestamp in signals.index:
            if timestamp in df.index:
                if signals.loc[timestamp] == 1:
                    current_pos = 1
                elif signals.loc[timestamp] == -1:
                    current_pos = -1
                position_indicator.loc[timestamp:] = current_pos
        
        ax3.fill_between(df.index, 0, position_indicator,
                        where=(position_indicator > 0), color='green', alpha=0.3, label='Long')
        ax3.fill_between(df.index, 0, position_indicator,
                        where=(position_indicator < 0), color='red', alpha=0.3, label='Short')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Position', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylim(-1.5, 1.5)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_equity_curve(self,
                         equity_curve: pd.DataFrame,
                         save_path: str = None) -> plt.Figure:
        """
        Plot equity curve and drawdown.
        
        Args:
            equity_curve: Equity curve DataFrame
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8),
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot equity curve
        ax1.plot(equity_curve['timestamp'], equity_curve['equity'],
                label='Equity', linewidth=2, color='blue')
        
        # Add peak equity line
        peak = equity_curve['equity'].expanding().max()
        ax1.plot(equity_curve['timestamp'], peak,
                label='Peak Equity', linewidth=1, color='gray', linestyle='--', alpha=0.7)
        
        # Fill area under equity curve
        ax1.fill_between(equity_curve['timestamp'], 0, equity_curve['equity'],
                        alpha=0.1, color='blue')
        
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        drawdown = (equity_curve['equity'] - peak) / peak * 100
        ax2.fill_between(equity_curve['timestamp'], 0, drawdown,
                        color='red', alpha=0.3)
        ax2.plot(equity_curve['timestamp'], drawdown,
                color='red', linewidth=1)
        
        ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_returns_distribution(self,
                                 returns: pd.Series,
                                 save_path: str = None) -> plt.Figure:
        """
        Plot returns distribution.
        
        Args:
            returns: Returns series
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(returns.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(x=returns.mean(), color='green', linestyle='--', alpha=0.7, label='Mean')
        ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_metrics(self,
                                metrics: Dict[str, Any],
                                save_path: str = None) -> plt.Figure:
        """
        Plot performance metrics summary.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Key metrics table
        key_metrics = {
            'Total Return': f"{metrics.get('total_return', 0)*100:.2f}%",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Max Drawdown': f"{metrics.get('max_drawdown', 0)*100:.2f}%",
            'Win Rate': f"{metrics.get('win_rate', 0)*100:.2f}%",
            'Total Trades': metrics.get('total_trades', 0),
            'Avg Trade P&L': f"${metrics.get('avg_trade_pnl', 0):.2f}"
        }
        
        ax = axes[0, 0]
        ax.axis('tight')
        ax.axis('off')
        table_data = [[k, v] for k, v in key_metrics.items()]
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        ax.set_title('Key Metrics', fontsize=14, fontweight='bold')
        
        # Win/Loss breakdown
        ax = axes[0, 1]
        wins = metrics.get('winning_trades', 0)
        losses = metrics.get('losing_trades', 0)
        if wins + losses > 0:
            sizes = [wins, losses]
            labels = [f'Wins ({wins})', f'Losses ({losses})']
            colors = ['green', 'red']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Win/Loss Breakdown', fontsize=14, fontweight='bold')
        
        # Cost breakdown
        ax = axes[1, 0]
        fees = metrics.get('total_fees', 0)
        slippage = metrics.get('total_slippage', 0)
        if fees + slippage > 0:
            sizes = [fees, slippage]
            labels = [f'Fees (${fees:.2f})', f'Slippage (${slippage:.2f})']
            colors = ['orange', 'purple']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Trading Costs Breakdown', fontsize=14, fontweight='bold')
        
        # Risk metrics
        ax = axes[1, 1]
        risk_metrics = {
            'Sharpe': metrics.get('sharpe_ratio', 0),
            'Sortino': metrics.get('sortino_ratio', 0),
            'Calmar': metrics.get('calmar_ratio', 0),
        }
        ax.bar(risk_metrics.keys(), risk_metrics.values(), color=['blue', 'green', 'orange'])
        ax.set_title('Risk-Adjusted Returns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ratio', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
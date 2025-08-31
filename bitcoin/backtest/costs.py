"""Cost models for slippage and fees."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


class CostModel:
    """Calculate trading costs including slippage and fees."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cost model.
        
        Args:
            config: Cost configuration
        """
        self.fee_bps = config.get('fee_bps', 5)
        self.slippage_config = config.get('slippage', {})
        self.slippage_mode = self.slippage_config.get('mode', 'powerlaw')
        
        # Power law parameters
        self.powerlaw_k = self.slippage_config.get('powerlaw_k', 1.5)
        self.powerlaw_c = self.slippage_config.get('powerlaw_c', 0.0005)
    
    def calculate_fee(self, trade_value: float) -> float:
        """
        Calculate trading fee.
        
        Args:
            trade_value: Absolute value of trade
            
        Returns:
            Fee amount
        """
        return abs(trade_value) * self.fee_bps / 10000
    
    def calculate_slippage_orderbook(self,
                                    orderbook_row: pd.Series,
                                    size: float,
                                    side: str) -> Tuple[float, float]:
        """
        Calculate slippage using orderbook data.
        
        Args:
            orderbook_row: Orderbook snapshot
            size: Trade size
            side: 'buy' or 'sell'
            
        Returns:
            (avg_price, slippage_bps)
        """
        if side == 'buy':
            # Start from best ask
            prices = []
            sizes = []
            
            # Parse ask levels
            for i in range(10):  # Assume max 10 levels
                price_col = f'ask_px_{i}' if i > 0 else 'ask_px'
                size_col = f'ask_sz_{i}' if i > 0 else 'ask_sz'
                
                if price_col in orderbook_row and size_col in orderbook_row:
                    prices.append(orderbook_row[price_col])
                    sizes.append(orderbook_row[size_col])
                else:
                    break
            
            if not prices:
                return orderbook_row.get('ask_px', orderbook_row.get('close', 0)), 0
            
        else:  # sell
            # Start from best bid
            prices = []
            sizes = []
            
            # Parse bid levels
            for i in range(10):
                price_col = f'bid_px_{i}' if i > 0 else 'bid_px'
                size_col = f'bid_sz_{i}' if i > 0 else 'bid_sz'
                
                if price_col in orderbook_row and size_col in orderbook_row:
                    prices.append(orderbook_row[price_col])
                    sizes.append(orderbook_row[size_col])
                else:
                    break
            
            if not prices:
                return orderbook_row.get('bid_px', orderbook_row.get('close', 0)), 0
        
        # Calculate average execution price
        remaining_size = abs(size)
        executed_value = 0
        executed_size = 0
        
        for price, available_size in zip(prices, sizes):
            if remaining_size <= 0:
                break
            
            exec_size = min(remaining_size, available_size)
            executed_value += exec_size * price
            executed_size += exec_size
            remaining_size -= exec_size
        
        if executed_size > 0:
            avg_price = executed_value / executed_size
        else:
            avg_price = prices[0] if prices else 0
        
        # Calculate slippage
        mid_price = (orderbook_row.get('bid_px', avg_price) + 
                    orderbook_row.get('ask_px', avg_price)) / 2
        slippage_bps = abs(avg_price - mid_price) / mid_price * 10000 if mid_price > 0 else 0
        
        return avg_price, slippage_bps
    
    def calculate_slippage_powerlaw(self,
                                   price: float,
                                   size: float,
                                   volume: float,
                                   side: str) -> Tuple[float, float]:
        """
        Calculate slippage using power law model.
        
        Args:
            price: Current price
            size: Trade size
            volume: Recent average volume
            side: 'buy' or 'sell'
            
        Returns:
            (execution_price, slippage_bps)
        """
        if volume <= 0:
            participation_rate = 0.1  # Default if no volume
        else:
            participation_rate = abs(size) / volume
        
        # Power law impact: impact = c * (participation_rate)^k
        price_impact = self.powerlaw_c * (participation_rate ** self.powerlaw_k)
        
        # Apply impact based on side
        if side == 'buy':
            execution_price = price * (1 + price_impact)
        else:
            execution_price = price * (1 - price_impact)
        
        slippage_bps = abs(execution_price - price) / price * 10000
        
        return execution_price, slippage_bps
    
    def calculate_total_cost(self,
                           price: float,
                           size: float,
                           side: str,
                           volume: float = None,
                           orderbook_row: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate total trading cost.
        
        Args:
            price: Base price
            size: Trade size
            side: 'buy' or 'sell'
            volume: Recent volume (for powerlaw model)
            orderbook_row: Orderbook data (for orderbook model)
            
        Returns:
            Dictionary with cost breakdown
        """
        # Calculate slippage
        if self.slippage_mode == 'orderbook' and orderbook_row is not None:
            exec_price, slippage_bps = self.calculate_slippage_orderbook(
                orderbook_row, size, side
            )
        else:
            # Use powerlaw model
            if volume is None:
                volume = abs(size) * 10  # Default assumption
            exec_price, slippage_bps = self.calculate_slippage_powerlaw(
                price, size, volume, side
            )
        
        # Calculate costs
        trade_value = abs(size * exec_price)
        fee = self.calculate_fee(trade_value)
        slippage_cost = abs(exec_price - price) * abs(size)
        total_cost = fee + slippage_cost
        
        return {
            'execution_price': exec_price,
            'slippage_bps': slippage_bps,
            'slippage_cost': slippage_cost,
            'fee': fee,
            'total_cost': total_cost,
            'cost_bps': total_cost / trade_value * 10000 if trade_value > 0 else 0
        }
    
    def estimate_round_trip_cost(self,
                                price: float,
                                size: float,
                                volume: float = None) -> float:
        """
        Estimate round-trip trading cost.
        
        Args:
            price: Current price
            size: Position size
            volume: Average volume
            
        Returns:
            Estimated round-trip cost in basis points
        """
        # Entry cost
        entry_costs = self.calculate_total_cost(price, size, 'buy', volume)
        
        # Exit cost (assume same conditions)
        exit_costs = self.calculate_total_cost(price, size, 'sell', volume)
        
        total_value = abs(size * price) * 2  # Round trip
        total_cost = entry_costs['total_cost'] + exit_costs['total_cost']
        
        return total_cost / total_value * 10000 if total_value > 0 else 0
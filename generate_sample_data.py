"""Generate realistic sample Bitcoin OHLCV data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_realistic_ohlcv(start_date='2023-01-01', 
                            end_date='2024-06-30',
                            freq='1min',
                            initial_price=40000):
    """Generate realistic OHLCV data with trend, volatility clustering, and market microstructure."""
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_periods = len(date_range)
    
    # Initialize arrays
    prices = np.zeros(n_periods)
    opens = np.zeros(n_periods)
    highs = np.zeros(n_periods)
    lows = np.zeros(n_periods)
    closes = np.zeros(n_periods)
    volumes = np.zeros(n_periods)
    
    # Price process parameters
    base_vol = 0.0002  # Base volatility per minute
    vol_persistence = 0.95
    trend_strength = 0.00001
    mean_reversion = 0.001
    
    # Volume parameters
    base_volume = 100
    volume_vol = 0.3
    
    # Initialize
    prices[0] = initial_price
    current_vol = base_vol
    
    for i in range(1, n_periods):
        # Time-based effects
        hour = date_range[i].hour
        day_of_week = date_range[i].dayofweek
        
        # Intraday volatility pattern (higher during US/EU hours)
        hour_multiplier = 1.0
        if 13 <= hour <= 21:  # UTC: US market hours
            hour_multiplier = 1.5
        elif 8 <= hour <= 16:  # UTC: EU market hours  
            hour_multiplier = 1.3
        elif 0 <= hour <= 8:   # UTC: Asian market hours
            hour_multiplier = 1.1
        else:
            hour_multiplier = 0.7  # Low activity hours
        
        # Weekend effect (lower volume/volatility)
        weekend_multiplier = 0.5 if day_of_week >= 5 else 1.0
        
        # Volatility clustering (GARCH-like)
        vol_innovation = np.random.normal(0, 0.0001)
        current_vol = base_vol * 0.05 + vol_persistence * current_vol + 0.1 * vol_innovation**2
        current_vol = max(current_vol, base_vol * 0.1)  # Floor
        
        # Apply time effects to volatility
        effective_vol = current_vol * hour_multiplier * weekend_multiplier
        
        # Price innovation with trend and mean reversion
        price_change_trend = trend_strength * np.random.choice([-1, 1], p=[0.4, 0.6])
        mean_reversion_force = -mean_reversion * np.log(prices[i-1] / initial_price)
        
        # Random shock
        shock = np.random.normal(0, effective_vol)
        
        # Combine effects
        log_return = price_change_trend + mean_reversion_force + shock
        
        # Apply price update
        prices[i] = prices[i-1] * np.exp(log_return)
        
        # Generate OHLC from close-to-close
        # Open is previous close with small gap
        gap = np.random.normal(0, effective_vol * 0.1)
        opens[i] = prices[i-1] * np.exp(gap)
        
        # Intraday range based on volatility
        intraday_range = effective_vol * np.random.gamma(2, 1) * 2
        
        # High and low around the close
        high_offset = np.random.uniform(0, intraday_range)
        low_offset = -np.random.uniform(0, intraday_range)
        
        highs[i] = prices[i] * np.exp(high_offset)
        lows[i] = prices[i] * np.exp(low_offset)
        closes[i] = prices[i]
        
        # Ensure OHLC consistency
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
        
        # Volume generation
        # Higher volume during high volatility and price moves
        vol_factor = current_vol / base_vol
        price_move_factor = abs(log_return) / effective_vol if effective_vol > 0 else 1
        
        volume_innovation = np.random.lognormal(0, volume_vol)
        volumes[i] = (base_volume * hour_multiplier * weekend_multiplier * 
                     vol_factor * price_move_factor * volume_innovation)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Remove any NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df


def generate_sample_orderbook(ohlcv_df, n_levels=5):
    """Generate sample L2 orderbook data."""
    
    orderbook_data = []
    
    # Sample every 100 minutes to reduce size
    sample_indices = range(0, len(ohlcv_df), 100)
    
    for idx in sample_indices:
        row = ohlcv_df.iloc[idx]
        mid_price = (row['high'] + row['low']) / 2
        
        # Generate spread
        spread_bps = np.random.gamma(2, 2) + 1  # 1-10 bps typical
        spread = mid_price * spread_bps / 10000
        
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generate orderbook levels
        orderbook_row = {
            'timestamp': row['timestamp'],
            'bid_px': best_bid,
            'ask_px': best_ask,
            'bid_sz': np.random.exponential(row['volume'] / 10),
            'ask_sz': np.random.exponential(row['volume'] / 10)
        }
        
        # Add deeper levels
        for i in range(1, n_levels):
            tick_size = mid_price * 0.0001  # 1bp tick size
            bid_price = best_bid - i * tick_size * np.random.uniform(1, 3)
            ask_price = best_ask + i * tick_size * np.random.uniform(1, 3)
            
            orderbook_row[f'bid_px_{i}'] = bid_price
            orderbook_row[f'ask_px_{i}'] = ask_price
            orderbook_row[f'bid_sz_{i}'] = np.random.exponential(row['volume'] / (10 * (i+1)))
            orderbook_row[f'ask_sz_{i}'] = np.random.exponential(row['volume'] / (10 * (i+1)))
        
        orderbook_data.append(orderbook_row)
    
    return pd.DataFrame(orderbook_data)


if __name__ == '__main__':
    print("Generating sample Bitcoin OHLCV data...")
    
    # Generate OHLCV data
    ohlcv_df = generate_realistic_ohlcv(
        start_date='2023-01-01',
        end_date='2024-06-30', 
        freq='1min',
        initial_price=40000
    )
    
    print(f"Generated {len(ohlcv_df):,} OHLCV records")
    print(f"Date range: {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")
    print(f"Price range: ${ohlcv_df['close'].min():.2f} to ${ohlcv_df['close'].max():.2f}")
    
    # Save OHLCV data
    ohlcv_df.to_csv('data/sample_ohlcv.csv', index=False)
    print("Saved OHLCV data to data/sample_ohlcv.csv")
    
    # Generate and save orderbook data
    print("\nGenerating sample orderbook data...")
    orderbook_df = generate_sample_orderbook(ohlcv_df)
    
    print(f"Generated {len(orderbook_df):,} orderbook snapshots")
    orderbook_df.to_csv('data/sample_orderbook.csv', index=False)
    print("Saved orderbook data to data/sample_orderbook.csv")
    
    # Display sample data
    print("\nSample OHLCV data:")
    print(ohlcv_df.head())
    
    print("\nSample orderbook data:")
    print(orderbook_df.head())
    
    print("\nData generation complete!")
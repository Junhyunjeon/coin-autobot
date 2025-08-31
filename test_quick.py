"""Quick test of the pipeline with smaller dataset."""

import pandas as pd
import numpy as np
from bitcoin.data import DataLoader, CUSUMEventDetector, TripleBarrierLabeler, FeatureEngineer
from bitcoin.models import GBDTModel
from bitcoin.backtest import BacktestEngine, MetricsCalculator

# Load small subset of data
print("Loading data...")
loader = DataLoader(tz='UTC')
df = loader.load_ohlcv('data/sample_ohlcv.csv')

# Use small subset for quick test
df_small = df.iloc[:10000]  # First 10k records
print(f"Using {len(df_small)} records for quick test")

# Split into train/test
split_idx = int(len(df_small) * 0.7)
train_data = df_small.iloc[:split_idx]
test_data = df_small.iloc[split_idx:]

print(f"Train: {len(train_data)} records, Test: {len(test_data)} records")

# Event detection
print("Detecting events...")
event_detector = CUSUMEventDetector(threshold=0.01)  # Higher threshold for fewer events
train_returns = train_data['close'].pct_change()
events = event_detector.detect_events(train_returns)
print(f"Found {len(events)} events")

# Triple barrier labeling
print("Labeling...")
labeler = TripleBarrierLabeler(pt_mult=2.0, sl_mult=1.0, max_holding_minutes=60)
labels_df = labeler.apply_triple_barrier(train_data, events)
print(f"Generated {len(labels_df)} labels")

# Feature engineering
print("Creating features...")
feature_engineer = FeatureEngineer({
    'returns_lags': [1, 5, 15],
    'vol_lookback': 20,
    'momentum_lookback': [10, 20]
})
train_features = feature_engineer.create_features(train_data)

# Get feature columns
feature_cols = [col for col in train_features.columns 
               if any(pattern in col for pattern in ['return_', 'volatility_', 'rsi_', 'macd'])]
print(f"Using {len(feature_cols)} features")

# Train simple model
print("Training model...")
model_config = {
    'lib': 'lightgbm',
    'params': {
        'n_estimators': 50,  # Fewer trees for speed
        'learning_rate': 0.1,
        'max_depth': 3,
        'verbose': -1
    },
    'thresholds': {
        'buy': 0.6,
        'sell': 0.6,
        'pass_band': [0.4, 0.6]
    }
}

gbdt_model = GBDTModel(model_config)
X_train, y_train = gbdt_model.prepare_data(train_features, labels_df, feature_cols)

if len(X_train) > 0:
    train_metrics = gbdt_model.train(X_train, y_train)
    print(f"Model trained. Validation accuracy: {train_metrics['val_accuracy']:.4f}")
    
    # Generate test signals
    print("Generating test signals...")
    test_features = feature_engineer.create_features(test_data)
    X_test = test_features[feature_cols].fillna(0).values
    test_probs = gbdt_model.predict(X_test)
    test_signals = pd.Series(
        gbdt_model.predict_signals(test_probs),
        index=test_features.index
    )
    
    print(f"Generated {(test_signals != 0).sum()} trading signals")
    
    # Simple position sizing
    position_sizes = test_signals * 0.1  # 10% position size
    
    # Run backtest
    print("Running backtest...")
    backtest_engine = BacktestEngine({
        'initial_capital': 10000,
        'costs': {'fee_bps': 5, 'slippage': {'mode': 'powerlaw', 'powerlaw_k': 1.5, 'powerlaw_c': 0.0005}}
    })
    
    results = backtest_engine.run_backtest(test_data, test_signals, position_sizes)
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_all_metrics(results)
    
    print("\n" + "="*40)
    print("QUICK TEST RESULTS")
    print("="*40)
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Total Fees: ${metrics['total_fees']:.2f}")
    print(f"Total Slippage: ${metrics['total_slippage']:.2f}")
    
    print("\nQuick test completed successfully!")
    
else:
    print("No training data available")
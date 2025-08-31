# Bitcoin Signal Pipeline

A comprehensive cryptocurrency signal generation and backtesting framework implementing sophisticated quantitative trading strategies.

## 🎯 Overview

This project implements a complete signal generation pipeline following the architecture:

**Event Detection (CUSUM)** → **Triple Barrier Labeling** → **Feature Engineering** → **GBDT Modeling** → **Meta-Labeling** → **Volatility Targeting** → **Scenario Policies** → **Guard Rules** → **Backtesting** → **Performance Analysis**

### Key Features

- 📊 **CUSUM-based event detection** for identifying significant market moves
- 🏷️ **Triple barrier labeling** for supervised learning with profit/loss/timeout exits
- 🔧 **Comprehensive feature engineering** (returns, volatility, momentum, volume, microstructure)
- 🌳 **Gradient boosting models** (LightGBM/XGBoost) for signal generation
- 🎯 **Meta-labeling** for signal filtering and confidence estimation  
- 📏 **Volatility targeting** for dynamic position sizing
- 📋 **Scenario-based policies** for regime-aware trading
- 🛡️ **Guard rules** for risk management (cooldown, limits, spread filtering)
- 💰 **Realistic cost modeling** (slippage, fees) with orderbook or power-law impact
- 📈 **Comprehensive backtesting** with detailed performance metrics and visualizations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd coin-autobot

# Install dependencies
pip install -r requirements.txt

# Install OpenMP (macOS)
brew install libomp
```

### Generate Sample Data

```bash
python generate_sample_data.py
```

This creates realistic Bitcoin OHLCV data with:
- Intraday volatility patterns
- Weekend effects
- Volatility clustering
- Market microstructure features

### Run Quick Test

```bash
python test_quick.py
```

### Run Full Pipeline

```bash
python -m bitcoin.cli.run --config configs/local.yaml --verbose
```

## 📁 Project Structure

```
bitcoin/
├── data/               # Data processing modules
│   ├── loader.py       # CSV/Parquet data loading
│   ├── events.py       # CUSUM event detection
│   ├── labeling.py     # Triple barrier labeling
│   └── features.py     # Feature engineering
├── models/             # ML models
│   ├── gbdt.py         # LightGBM/XGBoost wrapper
│   └── metalabel.py    # Meta-labeling model
├── sizing/             # Position sizing
│   └── vol_target.py   # Volatility targeting
├── policy/             # Trading policies
│   ├── scenarios.py    # Scenario-based rules
│   └── guard.py        # Risk management guards
├── backtest/           # Backtesting engine
│   ├── engine.py       # Event-driven backtest
│   ├── costs.py        # Slippage and fee models
│   ├── metrics.py      # Performance metrics
│   └── charts.py       # Visualization
└── cli/                # Command line interface
    └── run.py          # Main execution script

configs/
├── local.yaml          # Local configuration
scenarios.yaml          # Trading scenario definitions
data/
├── sample_ohlcv.csv    # Generated OHLCV data
└── sample_orderbook.csv # Generated orderbook data
reports/                # Generated reports
plots/                  # Generated charts
```

## ⚙️ Configuration

### configs/local.yaml

Key configuration sections:

```yaml
data:
  ohlcv_path: "data/sample_ohlcv.csv"
  orderbook_path: null  # Optional orderbook data

events:
  cusum_threshold: 0.005  # Event detection sensitivity

model:
  gbdt:
    lib: "lightgbm"
    thresholds:
      buy: 0.60    # Buy if P(up) >= 60%
      sell: 0.60   # Sell if P(down) >= 60%
      pass_band: [0.40, 0.60]  # Pass if uncertain

costs:
  fee_bps: 5  # 5bp trading fees
  slippage:
    mode: "powerlaw"  # or "orderbook"
    powerlaw_k: 1.5
    powerlaw_c: 0.0005
```

### scenarios.yaml

Define market regime-specific trading rules:

```yaml
scenarios:
  - name: "trend_only_low_spread"
    filters:
      regime: "trend"
      max_spread_bps: 8
    actions:
      allow: ["BUY", "SELL"]
      pass_when:
        prob_between: [0.40, 0.60]
```

## 🔬 Key Algorithms

### 1. CUSUM Event Detection

Detects significant price moves using cumulative sum statistics:

```python
detector = CUSUMEventDetector(threshold=0.005)
events = detector.detect_events(returns)
```

### 2. Triple Barrier Labeling

Labels events with profit-take, stop-loss, or timeout exits:

```python
labeler = TripleBarrierLabeler(pt_mult=2.0, sl_mult=1.0)
labels = labeler.apply_triple_barrier(df, events)
```

### 3. Meta-Labeling

Secondary model to filter primary signals:

```python
metalabel_model = MetaLabelModel(config)
filtered_signals = metalabel_model.filter_signals(primary_signals, metalabels)
```

### 4. Volatility Targeting

Dynamic position sizing based on volatility:

```python
vol_targeter = VolatilityTargeting(config)
sizes = vol_targeter.calculate_dynamic_sizing(df, signals)
```

## 📊 Output

### Performance Metrics

- **Return Metrics**: Total return, CAGR, Sharpe ratio, Sortino ratio
- **Risk Metrics**: Maximum drawdown, Calmar ratio, volatility
- **Trading Stats**: Win rate, profit factor, average trade P&L
- **Cost Analysis**: Total fees, slippage impact

### Generated Files

- `reports/metrics.json` - Detailed performance metrics
- `reports/tearsheet.md` - Human-readable performance report
- `plots/signal_overlay.png` - Price chart with trading signals
- `plots/equity_curve.png` - Equity and drawdown curves
- `plots/performance_metrics.png` - Performance summary dashboard

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run integration test:

```bash
python test_quick.py
```

## 🎛️ Advanced Usage

### Custom Feature Engineering

Add new features by extending the `FeatureEngineer` class:

```python
class CustomFeatureEngineer(FeatureEngineer):
    def _add_custom_features(self, df):
        # Add your custom features
        df['custom_indicator'] = ...
        return df
```

### Custom Slippage Models

Implement custom slippage models:

```python
def custom_slippage(price, size, volume, side):
    # Your custom slippage logic
    return execution_price, slippage_bps
```

### Walk-Forward Analysis

The pipeline supports walk-forward testing for realistic performance evaluation.

## 📈 Sample Results

Quick test on 10k samples:
- **Processing time**: ~30 seconds
- **Events detected**: 296 
- **Signals generated**: 2,378
- **Feature engineering**: 17 technical indicators
- **Model accuracy**: 60%

## 🔧 Troubleshooting

### Common Issues

1. **LightGBM OpenMP Error (macOS)**:
   ```bash
   brew install libomp
   export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
   ```

2. **Memory Issues with Large Datasets**:
   - Reduce CUSUM threshold to detect fewer events
   - Use data sampling or chunked processing
   - Increase system memory or use cloud instance

3. **Timezone Issues**:
   - Ensure all timestamps are UTC-aware
   - Check data loading timezone settings

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feat/new-feature`
3. Make changes and add tests
4. Commit: `git commit -m 'feat: add new feature'`
5. Push and create pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on scikit-learn, LightGBM, and pandas ecosystem
- Inspired by "Advances in Financial Machine Learning" by Marcos López de Prado
- Market microstructure concepts from academic literature
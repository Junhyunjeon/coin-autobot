"""CLI for running bitcoin signal pipeline."""

import click
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from bitcoin.data import DataLoader, CUSUMEventDetector, TripleBarrierLabeler, FeatureEngineer
from bitcoin.models import GBDTModel, MetaLabelModel
from bitcoin.sizing import VolatilityTargeting
from bitcoin.policy import ScenarioPolicy, GuardRules
from bitcoin.backtest import BacktestEngine, MetricsCalculator, ChartGenerator


@click.command()
@click.option('--config', '-c', default='configs/local.yaml', help='Path to config file')
@click.option('--mode', '-m', default='backtest', type=click.Choice(['backtest', 'train', 'predict']))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(config, mode, verbose):
    """Run Bitcoin signal pipeline."""
    
    # Load configuration
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if verbose:
        click.echo(f"Loading configuration from {config}")
        click.echo(f"Running in {mode} mode")
    
    # Initialize components
    click.echo("Initializing pipeline components...")
    
    # Data loader
    loader = DataLoader(tz=cfg['data'].get('tz', 'UTC'))
    
    # Load data
    click.echo(f"Loading data from {cfg['data']['ohlcv_path']}...")
    ohlcv_df = loader.load_ohlcv(cfg['data']['ohlcv_path'])
    
    # Load orderbook if available
    orderbook_df = None
    if cfg['data'].get('orderbook_path'):
        orderbook_df = loader.load_orderbook(cfg['data']['orderbook_path'])
    
    # Split data into train/test  
    train_start = pd.to_datetime(cfg['backtest'].get('train_start', '2023-01-01')).tz_localize('UTC')
    train_end = pd.to_datetime(cfg['backtest'].get('train_end', '2023-12-31')).tz_localize('UTC')
    test_start = pd.to_datetime(cfg['backtest'].get('test_start', '2024-01-01')).tz_localize('UTC')
    test_end = pd.to_datetime(cfg['backtest'].get('test_end', '2024-06-30')).tz_localize('UTC')
    
    train_data = ohlcv_df[train_start:train_end]
    test_data = ohlcv_df[test_start:test_end]
    
    if mode == 'train' or mode == 'backtest':
        click.echo("Detecting events with CUSUM...")
        
        # Event detection
        event_detector = CUSUMEventDetector(threshold=cfg['events']['cusum_threshold'])
        train_returns = train_data['close'].pct_change()
        train_events = event_detector.detect_events(train_returns)
        
        click.echo(f"Detected {len(train_events)} events in training data")
        
        # Triple barrier labeling
        click.echo("Applying triple barrier labeling...")
        labeler = TripleBarrierLabeler(
            pt_mult=cfg['labeling']['triple_barrier']['pt_mult'],
            sl_mult=cfg['labeling']['triple_barrier']['sl_mult'],
            max_holding_minutes=cfg['labeling']['triple_barrier']['max_holding_minutes']
        )
        
        labels_df = labeler.apply_triple_barrier(train_data, train_events)
        click.echo(f"Generated {len(labels_df)} labels")
        
        # Feature engineering
        click.echo("Engineering features...")
        feature_engineer = FeatureEngineer(cfg['features'])
        train_features = feature_engineer.create_features(train_data)
        
        # Get feature columns
        feature_cols = [col for col in train_features.columns 
                       if any(pattern in col for pattern in feature_engineer.get_feature_names())]
        
        # Train primary model
        click.echo("Training GBDT model...")
        gbdt_model = GBDTModel(cfg['model']['gbdt'])
        
        X_train, y_train = gbdt_model.prepare_data(train_features, labels_df, feature_cols)
        train_metrics = gbdt_model.train(X_train, y_train)
        
        click.echo(f"Model training complete. Validation accuracy: {train_metrics['val_accuracy']:.4f}")
        
        # Generate primary signals
        X_all = train_features[feature_cols].fillna(0).values
        train_probs = gbdt_model.predict(X_all)
        train_signals = pd.Series(
            gbdt_model.predict_signals(train_probs),
            index=train_features.index
        )
        
        # Train meta-label model
        if cfg['metalabel']['enable']:
            click.echo("Training meta-label model...")
            metalabel_model = MetaLabelModel(cfg['metalabel'])
            
            # Generate meta-labels
            metalabels_df = labeler.generate_metalabels(
                train_data, train_signals, train_events
            )
            
            if not metalabels_df.empty:
                meta_metrics = metalabel_model.train(
                    train_features, metalabels_df, feature_cols
                )
                click.echo(f"Meta-label model trained. F1 score: {meta_metrics.get('val_f1', 0):.4f}")
        else:
            metalabel_model = None
    
    if mode == 'backtest':
        click.echo("\nRunning backtest on test data...")
        
        # Generate features for test data
        test_features = feature_engineer.create_features(test_data)
        
        # Detect events in test data
        test_returns = test_data['close'].pct_change()
        test_events = event_detector.detect_events(test_returns)
        
        # Generate signals
        X_test = test_features[feature_cols].fillna(0).values
        test_probs = gbdt_model.predict(X_test)
        test_signals = pd.Series(
            gbdt_model.predict_signals(test_probs),
            index=test_features.index
        )
        
        # Apply meta-labeling
        if metalabel_model and metalabel_model.enabled:
            metalabels = metalabel_model.predict(test_features, test_signals, feature_cols)
            test_signals = metalabel_model.filter_signals(test_signals, metalabels)
            click.echo(f"Meta-labeling filtered {(metalabels == 0).sum()} signals")
        
        # Apply scenario policy
        if cfg['policy'].get('scenarios_path'):
            click.echo("Applying scenario policies...")
            scenario_policy = ScenarioPolicy(cfg['policy']['scenarios_path'])
            test_probs_series = pd.Series(test_probs.max(axis=1), index=test_features.index)
            test_signals = scenario_policy.filter_signals(
                test_signals, test_probs_series, test_data, orderbook_df
            )
        
        # Apply guard rules
        click.echo("Applying guard rules...")
        guard = GuardRules(cfg['guard'])
        
        filtered_signals = test_signals.copy()
        for timestamp in test_signals.index:
            if timestamp in test_data.index:
                vol = test_data.loc[:timestamp, 'close'].pct_change().rolling(60).std().iloc[-1]
                spread = 5  # Default spread
                
                filtered_signals.loc[timestamp] = guard.apply_guards(
                    test_signals.loc[timestamp],
                    timestamp,
                    spread_bps=spread,
                    current_vol=vol
                )
        
        # Position sizing
        click.echo("Calculating position sizes...")
        vol_targeter = VolatilityTargeting(cfg['sizing'])
        position_sizes = vol_targeter.calculate_dynamic_sizing(
            test_data, filtered_signals
        )
        
        # Run backtest
        click.echo("Running backtest engine...")
        backtest_engine = BacktestEngine({
            'initial_capital': 100000,
            'costs': cfg['costs']
        })
        
        results = backtest_engine.run_backtest(
            test_data, filtered_signals, position_sizes, orderbook_df
        )
        
        # Calculate metrics
        all_metrics = MetricsCalculator.calculate_all_metrics(results)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("BACKTEST RESULTS")
        click.echo("="*50)
        click.echo(f"Total Return: {all_metrics['total_return']*100:.2f}%")
        click.echo(f"Sharpe Ratio: {all_metrics['sharpe_ratio']:.2f}")
        click.echo(f"Max Drawdown: {all_metrics['max_drawdown']*100:.2f}%")
        click.echo(f"Win Rate: {all_metrics['win_rate']*100:.2f}%")
        click.echo(f"Total Trades: {all_metrics['total_trades']}")
        click.echo(f"CAGR: {all_metrics.get('cagr', 0)*100:.2f}%")
        click.echo(f"Calmar Ratio: {all_metrics.get('calmar_ratio', 0):.2f}")
        
        # Save results
        output_dir = Path(cfg['output']['report_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        click.echo(f"\nMetrics saved to {metrics_path}")
        
        # Generate tearsheet
        tearsheet_path = output_dir / 'tearsheet.md'
        generate_tearsheet(all_metrics, results, tearsheet_path)
        click.echo(f"Tearsheet saved to {tearsheet_path}")
        
        # Generate charts
        plot_dir = Path(cfg['output']['plot_dir'])
        plot_dir.mkdir(exist_ok=True)
        
        chart_gen = ChartGenerator()
        
        # Signal overlay chart
        fig = chart_gen.plot_signal_overlay(
            test_data, filtered_signals, results['trades'],
            save_path=plot_dir / 'signal_overlay.png'
        )
        click.echo(f"Signal overlay chart saved to {plot_dir / 'signal_overlay.png'}")
        
        # Equity curve
        if not results['equity_curve'].empty:
            fig = chart_gen.plot_equity_curve(
                results['equity_curve'],
                save_path=plot_dir / 'equity_curve.png'
            )
            click.echo(f"Equity curve chart saved to {plot_dir / 'equity_curve.png'}")
        
        # Performance metrics chart
        fig = chart_gen.plot_performance_metrics(
            all_metrics,
            save_path=plot_dir / 'performance_metrics.png'
        )
        click.echo(f"Performance metrics chart saved to {plot_dir / 'performance_metrics.png'}")
        
        click.echo("\nBacktest complete!")
    
    elif mode == 'predict':
        click.echo("Prediction mode not yet implemented")
    
    return 0


def generate_tearsheet(metrics, results, output_path):
    """Generate markdown tearsheet."""
    content = f"""# Bitcoin Signal Pipeline - Performance Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {metrics['total_return']*100:.2f}% |
| CAGR | {metrics.get('cagr', 0)*100:.2f}% |
| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} |
| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {metrics.get('calmar_ratio', 0):.2f} |
| Max Drawdown | {metrics['max_drawdown']*100:.2f}% |

## Trading Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {metrics['total_trades']} |
| Win Rate | {metrics['win_rate']*100:.2f}% |
| Average Win | ${metrics.get('avg_win', 0):.2f} |
| Average Loss | ${metrics.get('avg_loss', 0):.2f} |
| Profit Factor | {metrics.get('profit_factor', 0):.2f} |
| Expectancy | ${metrics.get('expectancy', 0):.2f} |

## Cost Analysis

| Cost Type | Amount |
|-----------|--------|
| Total Fees | ${metrics['total_fees']:.2f} |
| Total Slippage | ${metrics['total_slippage']:.2f} |
| Total Costs | ${metrics['total_fees'] + metrics['total_slippage']:.2f} |

## Risk Metrics

| Metric | Value |
|--------|-------|
| Annual Volatility | N/A |
| Downside Deviation | N/A |
| Value at Risk (95%) | N/A |
| Expected Shortfall | N/A |

## Trading Activity

- Annual Turnover: {metrics.get('annual_turnover', 0):.2f}x
- Average Holding Period: N/A
- Long/Short Ratio: N/A

---
*Report generated by Bitcoin Signal Pipeline*
"""
    
    with open(output_path, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    main()
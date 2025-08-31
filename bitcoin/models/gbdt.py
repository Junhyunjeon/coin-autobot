"""Gradient Boosting Decision Tree models for signal generation."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class GBDTModel:
    """Wrapper for LightGBM/XGBoost models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GBDT model.
        
        Args:
            config: Model configuration including library choice and parameters
        """
        self.lib = config.get('lib', 'lightgbm')
        self.params = config.get('params', {})
        self.thresholds = config.get('thresholds', {
            'buy': 0.60,
            'sell': 0.60,
            'pass_band': [0.40, 0.60]
        })
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        # Set default parameters based on library
        if self.lib == 'lightgbm':
            self._init_lightgbm()
        elif self.lib == 'xgboost':
            self._init_xgboost()
        else:
            raise ValueError(f"Unsupported library: {self.lib}")
    
    def _init_lightgbm(self):
        """Initialize LightGBM model with default parameters."""
        default_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(self.params)
        self.params = default_params
    
    def _init_xgboost(self):
        """Initialize XGBoost model with default parameters."""
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        default_params.update(self.params)
        self.params = default_params
    
    def prepare_data(self, 
                    features_df: pd.DataFrame,
                    labels_df: pd.DataFrame,
                    feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame with features
            labels_df: DataFrame with labels
            feature_cols: List of feature column names
            
        Returns:
            X, y arrays ready for training
        """
        # Merge features and labels
        data = features_df.merge(labels_df[['event_time', 'label']], 
                                left_index=True, 
                                right_on='event_time',
                                how='inner')
        
        # Filter feature columns that exist
        available_cols = [col for col in feature_cols if col in data.columns]
        
        X = data[available_cols].values
        y = data['label'].values
        
        # Convert labels to 0, 1, 2 for multiclass
        y = y + 1  # Convert -1,0,1 to 0,1,2
        
        return X, y
    
    def train(self, 
             X: np.ndarray, 
             y: np.ndarray,
             val_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the GBDT model.
        
        Args:
            X: Feature matrix
            y: Labels
            val_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Train model
        if self.lib == 'lightgbm':
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            self.feature_importance = self.model.feature_importances_
            
        elif self.lib == 'xgboost':
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            self.feature_importance = self.model.feature_importances_
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Get predictions for analysis
        val_probs = self.model.predict_proba(X_val)
        val_preds = self.predict_signals(val_probs)
        
        metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'n_training_samples': len(X_train),
            'n_validation_samples': len(X_val),
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities (n_samples, 3)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_signals(self, probs: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to trading signals.
        
        Args:
            probs: Probability matrix (n_samples, 3) for classes [sell, hold, buy]
            
        Returns:
            Signals: -1 (sell), 0 (pass), 1 (buy)
        """
        signals = np.zeros(len(probs))
        
        # Get probabilities for each class
        p_down = probs[:, 0]  # Class 0 (originally -1)
        p_neutral = probs[:, 1]  # Class 1 (originally 0)
        p_up = probs[:, 2]  # Class 2 (originally 1)
        
        # Apply thresholds
        buy_mask = p_up >= self.thresholds['buy']
        sell_mask = p_down >= self.thresholds['sell']
        
        # Check pass band
        pass_lower, pass_upper = self.thresholds['pass_band']
        pass_mask = (
            (p_up > pass_lower) & (p_up < pass_upper) &
            (p_down > pass_lower) & (p_down < pass_upper)
        )
        
        # Set signals
        signals[buy_mask] = 1
        signals[sell_mask] = -1
        signals[pass_mask] = 0
        
        # Handle conflicts (both buy and sell)
        conflict_mask = buy_mask & sell_mask
        signals[conflict_mask] = 0  # Pass on conflicts
        
        return signals
    
    def get_signal_distribution(self, signals: np.ndarray) -> Dict[str, float]:
        """Calculate distribution of signals."""
        total = len(signals)
        if total == 0:
            return {'buy': 0, 'sell': 0, 'pass': 0}
        
        return {
            'buy': (signals == 1).sum() / total,
            'sell': (signals == -1).sum() / total,
            'pass': (signals == 0).sum() / total
        }
    
    def save_model(self, path: str):
        """Save model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params,
            'thresholds': self.thresholds,
            'feature_importance': self.feature_importance,
            'lib': self.lib
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load model and scaler."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.params = model_data['params']
        self.thresholds = model_data['thresholds']
        self.feature_importance = model_data['feature_importance']
        self.lib = model_data['lib']
    
    def explain_prediction(self, X: np.ndarray, idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X: Feature matrix
            idx: Index of sample to explain
            
        Returns:
            Explanation dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled[idx:idx+1])[0]
        signal = self.predict_signals(probs.reshape(1, -1))[0]
        
        explanation = {
            'probabilities': {
                'sell': probs[0],
                'hold': probs[1],
                'buy': probs[2]
            },
            'signal': int(signal),
            'signal_name': {-1: 'SELL', 0: 'PASS', 1: 'BUY'}[signal],
            'confidence': float(np.max(probs))
        }
        
        return explanation
"""Meta-labeling model for signal filtering."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class MetaLabelModel:
    """Secondary model to filter primary signals."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize meta-label model.
        
        Args:
            config: Meta-label configuration
        """
        self.enabled = config.get('enable', True)
        self.params = config.get('params', {})
        
        # Default parameters for binary classification
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 31,
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
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0.5
    
    def prepare_metalabel_features(self,
                                  features_df: pd.DataFrame,
                                  primary_signals: pd.Series,
                                  feature_cols: list) -> pd.DataFrame:
        """
        Prepare features for meta-labeling.
        
        Args:
            features_df: Original features
            primary_signals: Primary model signals
            feature_cols: Feature column names
            
        Returns:
            Enhanced feature DataFrame
        """
        # Start with original features
        meta_features = features_df.copy()
        
        # Add primary signal as a feature
        meta_features['primary_signal'] = primary_signals
        meta_features['primary_signal_abs'] = np.abs(primary_signals)
        
        # Add signal context features
        meta_features['signal_change'] = primary_signals.diff()
        meta_features['signal_momentum'] = primary_signals.rolling(5).mean()
        meta_features['signal_volatility'] = primary_signals.rolling(10).std()
        
        # Count recent signals
        for window in [10, 30, 60]:
            meta_features[f'buy_count_{window}'] = (primary_signals == 1).rolling(window).sum()
            meta_features[f'sell_count_{window}'] = (primary_signals == -1).rolling(window).sum()
            meta_features[f'signal_ratio_{window}'] = (
                meta_features[f'buy_count_{window}'] / 
                (meta_features[f'sell_count_{window}'] + 1)  # Avoid division by zero
            )
        
        # Market regime features
        if 'volatility_20' in meta_features.columns:
            meta_features['high_vol_regime'] = (
                meta_features['volatility_20'] > meta_features['volatility_20'].rolling(100).mean()
            ).astype(int)
        
        if 'adx' in meta_features.columns:
            meta_features['trending_regime'] = (meta_features['adx'] > 25).astype(int)
        
        return meta_features
    
    def train(self,
             features_df: pd.DataFrame,
             metalabels_df: pd.DataFrame,
             feature_cols: list,
             val_split: float = 0.2) -> Dict[str, Any]:
        """
        Train meta-label model.
        
        Args:
            features_df: Feature DataFrame
            metalabels_df: Meta-labels DataFrame
            feature_cols: Feature column names
            val_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        if not self.enabled:
            return {'enabled': False}
        
        # Merge data
        data = features_df.merge(
            metalabels_df[['event_time', 'metalabel', 'primary_signal']],
            left_index=True,
            right_on='event_time',
            how='inner'
        )
        
        # Prepare meta features
        meta_features = self.prepare_metalabel_features(
            data.drop(['metalabel', 'event_time'], axis=1),
            data['primary_signal'],
            feature_cols
        )
        
        # Get available features
        available_cols = [col for col in meta_features.columns if not pd.isna(meta_features[col]).all()]
        
        X = meta_features[available_cols].fillna(0).values
        y = data['metalabel'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        self.params['scale_pos_weight'] = pos_weight
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Train model
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Calculate metrics
        val_probs = self.model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= self.threshold).astype(int)
        
        metrics = {
            'enabled': True,
            'train_accuracy': accuracy_score(y_train, self.model.predict(X_train)),
            'val_accuracy': accuracy_score(y_val, val_preds),
            'val_precision': precision_score(y_val, val_preds),
            'val_recall': recall_score(y_val, val_preds),
            'val_f1': f1_score(y_val, val_preds),
            'n_training_samples': len(X_train),
            'n_validation_samples': len(X_val),
            'pos_weight': pos_weight
        }
        
        return metrics
    
    def predict(self,
               features_df: pd.DataFrame,
               primary_signals: pd.Series,
               feature_cols: list) -> np.ndarray:
        """
        Predict meta-labels.
        
        Args:
            features_df: Feature DataFrame
            primary_signals: Primary model signals
            feature_cols: Feature column names
            
        Returns:
            Meta-label predictions (0: ignore, 1: take signal)
        """
        if not self.enabled or self.model is None:
            # If disabled or not trained, accept all signals
            return np.ones(len(features_df))
        
        # Prepare meta features
        meta_features = self.prepare_metalabel_features(
            features_df,
            primary_signals,
            feature_cols
        )
        
        # Get available features
        available_cols = [col for col in meta_features.columns if not pd.isna(meta_features[col]).all()]
        
        X = meta_features[available_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities and convert to binary
        probs = self.model.predict_proba(X_scaled)[:, 1]
        metalabels = (probs >= self.threshold).astype(int)
        
        return metalabels
    
    def filter_signals(self,
                      primary_signals: pd.Series,
                      metalabels: np.ndarray) -> pd.Series:
        """
        Filter primary signals using meta-labels.
        
        Args:
            primary_signals: Primary model signals
            metalabels: Meta-label predictions
            
        Returns:
            Filtered signals
        """
        filtered_signals = primary_signals.copy()
        
        # Set signal to 0 (PASS) where metalabel is 0 (ignore)
        mask = metalabels == 0
        filtered_signals[mask] = 0
        
        return filtered_signals
    
    def calculate_filtering_stats(self,
                                 primary_signals: pd.Series,
                                 filtered_signals: pd.Series) -> Dict[str, Any]:
        """Calculate statistics on signal filtering."""
        total_signals = (primary_signals != 0).sum()
        filtered_out = (primary_signals != 0) & (filtered_signals == 0)
        
        stats = {
            'total_primary_signals': int(total_signals),
            'signals_filtered_out': int(filtered_out.sum()),
            'filter_rate': float(filtered_out.sum() / total_signals) if total_signals > 0 else 0,
            'remaining_buy_signals': int((filtered_signals == 1).sum()),
            'remaining_sell_signals': int((filtered_signals == -1).sum()),
            'remaining_total_signals': int((filtered_signals != 0).sum())
        }
        
        return stats
    
    def save_model(self, path: str):
        """Save meta-label model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params,
            'threshold': self.threshold,
            'enabled': self.enabled
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load meta-label model."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.params = model_data['params']
        self.threshold = model_data['threshold']
        self.enabled = model_data['enabled']
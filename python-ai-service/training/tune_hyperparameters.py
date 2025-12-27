#!/usr/bin/env python3
"""
P2.1: Optuna-based Hyperparameter Optimization for AI Trading Models

This script provides automated hyperparameter tuning for:
1. LSTM+Transformer regressor
2. GBM models (XGBoost, LightGBM)
3. Quantile regressor

Uses time-series aware cross-validation to prevent look-ahead bias.

Usage:
    python training/tune_hyperparameters.py --symbol AAPL --model regressor --trials 100
    python training/tune_hyperparameters.py --symbol AAPL --model gbm --trials 50
    python training/tune_hyperparameters.py --symbol AAPL --model all --trials 100
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rolling_cv import create_time_series_cv, RollingWindowCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaHyperparameterTuner:
    """Unified hyperparameter tuner for all model types."""
    
    def __init__(
        self,
        symbol: str,
        model_type: str = "regressor",
        n_trials: int = 100,
        cv_method: str = "rolling",
        n_splits: int = 5,
        metric: str = "sharpe",  # 'sharpe', 'mse', 'mae', 'directional_accuracy'
        study_name: Optional[str] = None,
        storage: Optional[str] = None,  # e.g., 'sqlite:///optuna.db' for persistence
        seed: int = 42,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            symbol: Stock symbol for data loading
            model_type: 'regressor', 'gbm', 'quantile', or 'all'
            n_trials: Number of optimization trials
            cv_method: 'rolling', 'expanding', or 'purged'
            n_splits: Number of CV splits
            metric: Optimization metric
            study_name: Name for the Optuna study
            storage: Database URL for study persistence
            seed: Random seed
        """
        self.symbol = symbol
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.metric = metric
        self.seed = seed
        
        self.study_name = study_name or f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        
        # Results storage
        self.results_dir = Path(__file__).parent.parent / "training_logs" / "optuna"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Sampler with seed for reproducibility
        self.sampler = TPESampler(seed=seed, n_startup_trials=10)
        
        # Pruner to stop unpromising trials early
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        
        # Data cache
        self._X = None
        self._y = None
        self._dates = None
        self._prices = None
    
    def load_data(self) -> tuple:
        """Load and preprocess data for the symbol."""
        if self._X is not None:
            return self._X, self._y, self._dates, self._prices
        
        # Import data modules
        from data.data_fetcher import fetch_stock_data
        from data.feature_engineer import engineer_features
        
        logger.info(f"Loading data for {self.symbol}...")
        
        # Get raw data
        df = fetch_stock_data(self.symbol, period="5y")
        if df is None or df.empty:
            raise ValueError(f"Failed to load data for {self.symbol}")
        
        # Engineer features
        df = engineer_features(df, symbol=self.symbol, include_sentiment=False)
        
        # Create forward return target if not exists
        if 'target_1d' not in df.columns:
            df['target_1d'] = df['Close'].pct_change().shift(-1)
        
        # Drop NaN
        df = df.dropna()
        
        # Extract features and targets
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Date']
        exclude_cols += [c for c in df.columns if c.startswith('target_') or c.endswith('_scaled') or c.endswith('_clipped')]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        self._X = df[feature_cols].values
        self._y = df['target_1d'].values if 'target_1d' in df.columns else df['forward_return'].values
        self._dates = df.index.to_numpy() if hasattr(df.index, 'to_numpy') else np.array(df.index)
        self._prices = df['Close'].values
        
        logger.info(f"Loaded {len(self._y)} samples with {len(feature_cols)} features")
        
        return self._X, self._y, self._dates, self._prices
    
    def get_cv_splits(self, X: np.ndarray, y: np.ndarray) -> list:
        """Get cross-validation splits using time-series aware method."""
        n_samples = len(X)
        # Calculate test_size as integer (60 days default, or 20% of data if rolling)
        if self.cv_method == 'rolling':
            test_size = max(60, int(n_samples * 0.2 / self.n_splits))
        else:
            test_size = 60  # Default 60 trading days (~3 months)
        
        cv = create_time_series_cv(
            cv_type=self.cv_method,
            n_splits=self.n_splits,
            test_size=test_size,
            min_train_size=252  # At least 1 year of training
        )
        return list(cv.split(X, y))
    
    def compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """Compute the optimization metric."""
        if self.metric == 'mse':
            return -mean_squared_error(y_true, y_pred)  # Negative for maximization
        elif self.metric == 'mae':
            return -mean_absolute_error(y_true, y_pred)  # Negative for maximization
        elif self.metric == 'directional_accuracy':
            # Percentage of correct direction predictions
            correct = np.sum(np.sign(y_true) == np.sign(y_pred))
            return correct / len(y_true)
        elif self.metric == 'sharpe':
            # Compute Sharpe ratio from strategy returns
            positions = np.sign(y_pred)  # Simple directional positions
            returns = positions * y_true
            if np.std(returns) == 0:
                return 0.0
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            return sharpe
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def suggest_regressor_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for LSTM+Transformer regressor."""
        return {
            # Architecture
            'sequence_length': trial.suggest_int('sequence_length', 20, 120, step=10),
            'lstm_units': trial.suggest_categorical('lstm_units', [32, 64, 128, 256]),
            'num_lstm_layers': trial.suggest_int('num_lstm_layers', 1, 3),
            'num_transformer_heads': trial.suggest_categorical('num_transformer_heads', [2, 4, 8]),
            'transformer_ff_dim': trial.suggest_categorical('transformer_ff_dim', [64, 128, 256]),
            'dense_units': trial.suggest_categorical('dense_units', [32, 64, 128]),
            
            # Regularization
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05),
            'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True),
            
            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 30, 150, step=10),
            
            # Loss
            'loss_type': trial.suggest_categorical('loss_type', ['huber', 'mse', 'mae']),
            'huber_delta': trial.suggest_float('huber_delta', 0.01, 1.0, log=True),
        }
    
    def suggest_gbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for GBM models (XGBoost + LightGBM)."""
        model_type = trial.suggest_categorical('gbm_type', ['xgboost', 'lightgbm'])
        
        common_params = {
            'gbm_type': model_type,
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        
        if model_type == 'lightgbm':
            common_params.update({
                'num_leaves': trial.suggest_int('num_leaves', 10, 256),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'huber_delta': trial.suggest_float('huber_delta', 0.001, 0.1, log=True),  # P0.2 fix range
            })
        else:  # xgboost
            common_params.update({
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5),
            })
        
        return common_params
    
    def suggest_quantile_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for quantile regressor."""
        return {
            'sequence_length': trial.suggest_int('sequence_length', 20, 120, step=10),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 50, 200, step=25),
            'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],  # Fixed quantiles
        }
    
    def create_regressor_model(self, params: Dict[str, Any], input_shape: tuple):
        """Create LSTM+Transformer model with given hyperparameters."""
        import tensorflow as tf
        from tensorflow.keras import layers, Model, regularizers
        
        seq_len, n_features = input_shape
        
        # Input
        inputs = layers.Input(shape=(seq_len, n_features))
        x = inputs
        
        # LSTM layers
        for i in range(params['num_lstm_layers']):
            return_sequences = (i < params['num_lstm_layers'] - 1)
            x = layers.LSTM(
                params['lstm_units'],
                return_sequences=return_sequences or (params['num_transformer_heads'] > 0),
                kernel_regularizer=regularizers.l2(params['l2_reg']),
                dropout=params['dropout_rate'],
                recurrent_dropout=params['dropout_rate'] * 0.5
            )(x)
        
        # Transformer attention (if sequence output)
        if len(x.shape) == 3:
            attn_output = layers.MultiHeadAttention(
                num_heads=params['num_transformer_heads'],
                key_dim=params['lstm_units'] // params['num_transformer_heads'],
                dropout=params['dropout_rate']
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(
            params['dense_units'],
            activation='relu',
            kernel_regularizer=regularizers.l2(params['l2_reg'])
        )(x)
        x = layers.Dropout(params['dropout_rate'])(x)
        
        # Output
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = Model(inputs, outputs)
        
        # Compile
        if params['loss_type'] == 'huber':
            loss = tf.keras.losses.Huber(delta=params['huber_delta'])
        elif params['loss_type'] == 'mse':
            loss = 'mse'
        else:
            loss = 'mae'
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=loss,
            metrics=['mae']
        )
        
        return model
    
    def create_gbm_model(self, params: Dict[str, Any]):
        """Create GBM model with given hyperparameters."""
        if params['gbm_type'] == 'lightgbm':
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_samples=params['min_child_samples'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                num_leaves=params.get('num_leaves', 31),
                reg_alpha=params.get('reg_alpha', 0.0),
                reg_lambda=params.get('reg_lambda', 0.0),
                objective='huber',
                huber_delta=params.get('huber_delta', 0.01),  # P0.2 fix
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1
            )
        else:  # xgboost
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_samples=params['min_child_samples'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params.get('xgb_reg_alpha', 0.0),
                reg_lambda=params.get('xgb_reg_lambda', 0.0),
                gamma=params.get('gamma', 0),
                random_state=self.seed,
                n_jobs=-1,
                verbosity=0
            )
        return model
    
    def objective_regressor(self, trial: optuna.Trial) -> float:
        """Optuna objective for regressor tuning."""
        params = self.suggest_regressor_params(trial)
        X, y, dates, prices = self.load_data()
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        seq_len = params['sequence_length']
        X_seq = []
        y_seq = []
        for i in range(seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i-seq_len:i])
            y_seq.append(y[i])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # CV evaluation
        cv_splits = self.get_cv_splits(X_seq, y_seq)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]
            
            # Create and train model
            model = self.create_regressor_model(params, X_train.shape[1:])
            
            # Early stopping
            import tensorflow as tf
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_val, verbose=0).flatten()
            score = self.compute_metric(y_val, y_pred)
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Clear session to free memory
            tf.keras.backend.clear_session()
        
        return np.mean(scores)
    
    def objective_gbm(self, trial: optuna.Trial) -> float:
        """Optuna objective for GBM tuning."""
        params = self.suggest_gbm_params(trial)
        X, y, dates, prices = self.load_data()
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # CV evaluation
        cv_splits = self.get_cv_splits(X_scaled, y)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = self.create_gbm_model(params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            score = self.compute_metric(y_val, y_pred)
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run the hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization for {self.model_type}")
        logger.info(f"Study: {self.study_name}, Trials: {self.n_trials}, Metric: {self.metric}")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction='maximize',
            load_if_exists=True
        )
        
        # Select objective
        if self.model_type == 'regressor':
            objective = self.objective_regressor
        elif self.model_type == 'gbm':
            objective = self.objective_gbm
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        # Get results
        best_trial = study.best_trial
        results = {
            'study_name': self.study_name,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'metric': self.metric,
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'n_trials': len(study.trials),
            'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save results
        results_path = self.results_dir / f"{self.study_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimization Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Best {self.metric}: {best_trial.value:.6f}")
        logger.info(f"Completed trials: {results['n_completed']}/{results['n_trials']}")
        logger.info(f"Pruned trials: {results['n_pruned']}")
        logger.info(f"\nBest parameters:")
        for k, v in best_trial.params.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"\nResults saved to: {results_path}")
        
        # Generate optimization visualization
        self._plot_optimization_history(study)
        
        return results
    
    def _plot_optimization_history(self, study: optuna.Study):
        """Generate optimization visualization plots."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Optimization history
            ax = axes[0, 0]
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            values = [t.value for t in trials]
            ax.plot(range(len(values)), values, 'b-', alpha=0.5, label='Trial value')
            ax.plot(range(len(values)), np.maximum.accumulate(values), 'r-', linewidth=2, label='Best so far')
            ax.set_xlabel('Trial')
            ax.set_ylabel(self.metric.capitalize())
            ax.set_title('Optimization History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Parameter importance
            ax = axes[0, 1]
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:10]  # Top 10
                values = [importance[p] for p in params]
                ax.barh(params, values)
                ax.set_xlabel('Importance')
                ax.set_title('Parameter Importance')
            except Exception:
                ax.text(0.5, 0.5, 'Insufficient trials\nfor importance analysis', 
                       ha='center', va='center', transform=ax.transAxes)
            
            # Plot 3: Parallel coordinate
            ax = axes[1, 0]
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) > 1:
                df_trials = pd.DataFrame([{**t.params, 'value': t.value} for t in completed])
                # Normalize
                for col in df_trials.columns:
                    if col != 'value' and df_trials[col].dtype in [np.float64, np.int64]:
                        min_val, max_val = df_trials[col].min(), df_trials[col].max()
                        if max_val > min_val:
                            df_trials[col] = (df_trials[col] - min_val) / (max_val - min_val)
                
                # Color by value
                cmap = plt.cm.viridis
                norm = plt.Normalize(df_trials['value'].min(), df_trials['value'].max())
                
                for idx, row in df_trials.iterrows():
                    color = cmap(norm(row['value']))
                    numeric_cols = [c for c in df_trials.columns if c != 'value' and df_trials[c].dtype in [np.float64, np.int64]][:8]
                    ax.plot(range(len(numeric_cols)), [row[c] for c in numeric_cols], alpha=0.5, color=color)
                
                ax.set_xticks(range(len(numeric_cols)))
                ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                ax.set_ylabel('Normalized Value')
                ax.set_title('Parallel Coordinates (colored by objective)')
            else:
                ax.text(0.5, 0.5, 'Insufficient trials', ha='center', va='center', transform=ax.transAxes)
            
            # Plot 4: Value distribution
            ax = axes[1, 1]
            values = [t.value for t in completed if t.value is not None]
            if values:
                ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
                ax.axvline(max(values), color='r', linestyle='--', label=f'Best: {max(values):.4f}')
                ax.set_xlabel(self.metric.capitalize())
                ax.set_ylabel('Count')
                ax.set_title('Objective Value Distribution')
                ax.legend()
            
            plt.tight_layout()
            
            plot_path = self.results_dir / f"{self.study_name}_optimization.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Optimization plot saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate optimization plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--model', type=str, default='gbm', choices=['regressor', 'gbm', 'quantile', 'all'],
                       help='Model type to tune')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--cv', type=str, default='rolling', choices=['rolling', 'expanding', 'purged'],
                       help='Cross-validation method')
    parser.add_argument('--splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--metric', type=str, default='sharpe', 
                       choices=['sharpe', 'mse', 'mae', 'directional_accuracy'],
                       help='Optimization metric')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    models_to_tune = [args.model] if args.model != 'all' else ['regressor', 'gbm']
    
    all_results = {}
    for model_type in models_to_tune:
        logger.info(f"\n{'='*60}")
        logger.info(f"Tuning {model_type.upper()} for {args.symbol}")
        logger.info(f"{'='*60}")
        
        tuner = OptunaHyperparameterTuner(
            symbol=args.symbol,
            model_type=model_type,
            n_trials=args.trials,
            cv_method=args.cv,
            n_splits=args.splits,
            metric=args.metric,
            storage=args.storage,
            seed=args.seed,
        )
        
        results = tuner.run_optimization()
        all_results[model_type] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for model_type, results in all_results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Best {args.metric}: {results['best_value']:.6f}")
        print(f"  Best params: {results['best_params']}")


if __name__ == '__main__':
    main()

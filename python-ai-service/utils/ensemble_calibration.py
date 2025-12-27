#!/usr/bin/env python3
"""
P2.3: Ensemble Calibration Module

Implements probability calibration for ensemble model outputs:
1. Isotonic Regression calibration
2. Platt Scaling (sigmoid calibration)
3. Temperature Scaling
4. Calibration diagnostics (reliability diagrams)

This ensures that predicted probabilities reflect true outcome frequencies.

Usage:
    from utils.ensemble_calibration import CalibratedEnsemble
    
    calibrator = CalibratedEnsemble(method='isotonic')
    calibrator.fit(predicted_probs, actual_outcomes)
    calibrated_probs = calibrator.transform(new_predictions)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for probability outputs.
    
    Isotonic regression fits a non-decreasing step function to map
    raw scores to calibrated probabilities. It's non-parametric and
    makes minimal assumptions about the score distribution.
    """
    
    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Initialize the calibrator.
        
        Args:
            out_of_bounds: How to handle values outside training range
                          ('clip', 'nan', or 'raise')
        """
        self.out_of_bounds = out_of_bounds
        self.calibrator_ = None
        self.y_min_ = None
        self.y_max_ = None
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit the isotonic regression calibrator.
        
        Args:
            y_pred: Raw probability predictions
            y_true: Actual binary outcomes (0 or 1)
        
        Returns:
            self
        """
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()
        
        # Store bounds for transform
        self.y_min_ = y_pred.min()
        self.y_max_ = y_pred.max()
        
        # Fit isotonic regression
        self.calibrator_ = IsotonicRegression(
            out_of_bounds=self.out_of_bounds,
            y_min=0.0,
            y_max=1.0
        )
        self.calibrator_.fit(y_pred, y_true)
        
        return self
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform raw predictions to calibrated probabilities.
        
        Args:
            y_pred: Raw probability predictions
        
        Returns:
            Calibrated probabilities
        """
        if self.calibrator_ is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_pred = np.asarray(y_pred).flatten()
        return self.calibrator_.transform(y_pred)
    
    def fit_transform(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_pred, y_true)
        return self.transform(y_pred)


class PlattScaler:
    """
    Platt scaling (sigmoid calibration) for probability outputs.
    
    Fits a logistic regression to map raw scores to calibrated probabilities.
    Assumes the underlying score distribution can be calibrated with a
    simple sigmoid transformation.
    """
    
    def __init__(self, max_iter: int = 1000):
        """
        Initialize Platt scaler.
        
        Args:
            max_iter: Maximum iterations for logistic regression
        """
        self.max_iter = max_iter
        self.calibrator_ = None
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> 'PlattScaler':
        """
        Fit the Platt scaler.
        
        Args:
            y_pred: Raw probability predictions
            y_true: Actual binary outcomes
        
        Returns:
            self
        """
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        y_true = np.asarray(y_true).flatten()
        
        self.calibrator_ = LogisticRegression(
            max_iter=self.max_iter,
            solver='lbfgs'
        )
        self.calibrator_.fit(y_pred, y_true)
        
        return self
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform raw predictions to calibrated probabilities.
        
        Args:
            y_pred: Raw probability predictions
        
        Returns:
            Calibrated probabilities
        """
        if self.calibrator_ is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        return self.calibrator_.predict_proba(y_pred)[:, 1]
    
    def fit_transform(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_pred, y_true)
        return self.transform(y_pred)


class TemperatureScaler:
    """
    Temperature scaling for neural network probability outputs.
    
    Learns a single temperature parameter T to scale logits:
    p_calibrated = softmax(logits / T)
    
    For binary classification, this simplifies to:
    p_calibrated = sigmoid((logit(p) / T))
    """
    
    def __init__(self, init_temperature: float = 1.0, max_iter: int = 100):
        """
        Initialize temperature scaler.
        
        Args:
            init_temperature: Initial temperature value
            max_iter: Maximum optimization iterations
        """
        self.init_temperature = init_temperature
        self.max_iter = max_iter
        self.temperature_ = init_temperature
    
    def _logit(self, p: np.ndarray) -> np.ndarray:
        """Convert probability to logit."""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaler':
        """
        Fit temperature parameter using NLL minimization.
        
        Args:
            y_pred: Raw probability predictions
            y_true: Actual binary outcomes
        
        Returns:
            self
        """
        from scipy.optimize import minimize_scalar
        
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()
        
        logits = self._logit(y_pred)
        
        def nll(T):
            if T <= 0:
                return np.inf
            scaled_probs = self._sigmoid(logits / T)
            # Clip to avoid log(0)
            scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
            return -np.mean(
                y_true * np.log(scaled_probs) + 
                (1 - y_true) * np.log(1 - scaled_probs)
            )
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature_ = result.x
        
        return self
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform raw predictions using learned temperature.
        
        Args:
            y_pred: Raw probability predictions
        
        Returns:
            Calibrated probabilities
        """
        y_pred = np.asarray(y_pred).flatten()
        logits = self._logit(y_pred)
        return self._sigmoid(logits / self.temperature_)
    
    def fit_transform(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_pred, y_true)
        return self.transform(y_pred)


class CalibratedEnsemble:
    """
    Ensemble calibration for multi-model probability outputs.
    
    Calibrates multiple model outputs and optionally combines them
    with learned weights.
    """
    
    def __init__(
        self, 
        method: str = 'isotonic',
        combine_method: str = 'average',
        n_bins: int = 10,
    ):
        """
        Initialize calibrated ensemble.
        
        Args:
            method: Calibration method ('isotonic', 'platt', 'temperature')
            combine_method: How to combine multiple outputs ('average', 'weighted')
            n_bins: Number of bins for reliability diagram
        """
        self.method = method
        self.combine_method = combine_method
        self.n_bins = n_bins
        
        self.calibrators_: Dict[str, object] = {}
        self.weights_: Optional[np.ndarray] = None
        self.metrics_: Dict[str, Dict] = {}
        
        # Results directory
        self.results_dir = Path(__file__).parent.parent / "training_logs" / "calibration"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_calibrator(self):
        """Factory method for calibrator creation."""
        if self.method == 'isotonic':
            return IsotonicCalibrator()
        elif self.method == 'platt':
            return PlattScaler()
        elif self.method == 'temperature':
            return TemperatureScaler()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        learn_weights: bool = True
    ) -> 'CalibratedEnsemble':
        """
        Fit calibrators for each model's predictions.
        
        Args:
            predictions: Dict mapping model names to their predictions
            y_true: Actual binary outcomes
            learn_weights: Whether to learn combination weights
        
        Returns:
            self
        """
        y_true = np.asarray(y_true).flatten()
        
        for name, y_pred in predictions.items():
            y_pred = np.asarray(y_pred).flatten()
            
            # Create and fit calibrator
            calibrator = self._create_calibrator()
            calibrator.fit(y_pred, y_true)
            self.calibrators_[name] = calibrator
            
            # Compute calibration metrics
            y_calibrated = calibrator.transform(y_pred)
            self.metrics_[name] = self._compute_metrics(y_pred, y_calibrated, y_true)
            
            logger.info(f"Fitted calibrator for {name}: "
                       f"Brier before={self.metrics_[name]['brier_before']:.4f}, "
                       f"after={self.metrics_[name]['brier_after']:.4f}")
        
        # Learn combination weights if requested
        if learn_weights and len(predictions) > 1:
            self._learn_weights(predictions, y_true)
        else:
            # Equal weights
            self.weights_ = np.ones(len(predictions)) / len(predictions)
        
        return self
    
    def _compute_metrics(
        self, 
        y_pred_raw: np.ndarray, 
        y_pred_calibrated: np.ndarray, 
        y_true: np.ndarray
    ) -> Dict:
        """Compute calibration quality metrics."""
        return {
            'brier_before': brier_score_loss(y_true, y_pred_raw),
            'brier_after': brier_score_loss(y_true, y_pred_calibrated),
            'brier_improvement': (
                brier_score_loss(y_true, y_pred_raw) - 
                brier_score_loss(y_true, y_pred_calibrated)
            ),
            'log_loss_before': log_loss(y_true, np.clip(y_pred_raw, 1e-10, 1-1e-10)),
            'log_loss_after': log_loss(y_true, np.clip(y_pred_calibrated, 1e-10, 1-1e-10)),
        }
    
    def _learn_weights(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """Learn optimal combination weights using NLL minimization."""
        from scipy.optimize import minimize
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        # Get calibrated predictions
        calibrated = np.column_stack([
            self.calibrators_[name].transform(predictions[name])
            for name in model_names
        ])
        
        def nll(weights):
            weights = weights / np.sum(weights)  # Normalize
            combined = np.dot(calibrated, weights)
            combined = np.clip(combined, 1e-10, 1 - 1e-10)
            return -np.mean(
                y_true * np.log(combined) + 
                (1 - y_true) * np.log(1 - combined)
            )
        
        # Initial equal weights
        x0 = np.ones(n_models) / n_models
        
        # Optimize with constraint that weights sum to 1
        result = minimize(
            nll,
            x0,
            method='SLSQP',
            bounds=[(0.01, 1.0)] * n_models,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        self.weights_ = result.x
        logger.info(f"Learned combination weights: {dict(zip(model_names, self.weights_))}")
    
    def transform(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform and combine predictions from multiple models.
        
        Args:
            predictions: Dict mapping model names to their predictions
        
        Returns:
            Combined calibrated probabilities
        """
        if not self.calibrators_:
            raise ValueError("Calibrators not fitted. Call fit() first.")
        
        model_names = list(self.calibrators_.keys())
        
        # Calibrate each model's predictions
        calibrated = []
        for name in model_names:
            if name not in predictions:
                raise ValueError(f"Missing predictions for model: {name}")
            y_pred = np.asarray(predictions[name]).flatten()
            calibrated.append(self.calibrators_[name].transform(y_pred))
        
        calibrated = np.column_stack(calibrated)
        
        # Combine with weights
        if self.combine_method == 'weighted':
            combined = np.dot(calibrated, self.weights_)
        else:
            combined = np.mean(calibrated, axis=1)
        
        return combined
    
    def plot_reliability_diagram(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot reliability diagrams before and after calibration.
        
        Args:
            predictions: Dict mapping model names to their predictions
            y_true: Actual binary outcomes
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            n_models = len(predictions)
            fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
            
            if n_models == 1:
                axes = axes.reshape(2, 1)
            
            for idx, (name, y_pred) in enumerate(predictions.items()):
                y_pred = np.asarray(y_pred).flatten()
                y_true_arr = np.asarray(y_true).flatten()
                
                # Before calibration
                ax = axes[0, idx]
                prob_true, prob_pred = calibration_curve(
                    y_true_arr, y_pred, n_bins=self.n_bins, strategy='uniform'
                )
                ax.plot(prob_pred, prob_true, marker='o', label=name)
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title(f'{name} - Before Calibration')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # After calibration
                ax = axes[1, idx]
                if name in self.calibrators_:
                    y_calibrated = self.calibrators_[name].transform(y_pred)
                    prob_true, prob_pred = calibration_curve(
                        y_true_arr, y_calibrated, n_bins=self.n_bins, strategy='uniform'
                    )
                    ax.plot(prob_pred, prob_true, marker='o', label=f'{name} (calibrated)')
                    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.set_title(f'{name} - After {self.method.title()} Calibration')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Reliability diagram saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot reliability diagram: {e}")
    
    def get_calibration_report(self) -> pd.DataFrame:
        """Generate calibration metrics report."""
        if not self.metrics_:
            raise ValueError("No metrics available. Call fit() first.")
        
        rows = []
        for name, metrics in self.metrics_.items():
            row = {'model': name, **metrics}
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save(self, filepath: str):
        """Save calibrators and weights to file."""
        import pickle
        
        state = {
            'method': self.method,
            'combine_method': self.combine_method,
            'calibrators': self.calibrators_,
            'weights': self.weights_,
            'metrics': self.metrics_,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Calibrated ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CalibratedEnsemble':
        """Load calibrators and weights from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        ensemble = cls(
            method=state['method'],
            combine_method=state['combine_method']
        )
        ensemble.calibrators_ = state['calibrators']
        ensemble.weights_ = state['weights']
        ensemble.metrics_ = state['metrics']
        
        logger.info(f"Calibrated ensemble loaded from {filepath}")
        return ensemble


def calibrate_ensemble_predictions(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    method: str = 'isotonic',
    symbol: str = 'UNKNOWN',
    save_results: bool = True,
) -> Tuple[np.ndarray, CalibratedEnsemble]:
    """
    Convenience function for calibrating ensemble predictions.
    
    Args:
        predictions: Dict mapping model names to predictions
        y_true: Actual binary outcomes
        method: Calibration method
        symbol: Stock symbol for saving
        save_results: Whether to save results
    
    Returns:
        (calibrated_predictions, calibrator)
    """
    calibrator = CalibratedEnsemble(method=method)
    calibrator.fit(predictions, y_true)
    calibrated = calibrator.transform(predictions)
    
    if save_results:
        # Save calibrator
        save_path = calibrator.results_dir / f"{symbol}_{method}_calibrator.pkl"
        calibrator.save(str(save_path))
        
        # Plot reliability diagram
        plot_path = calibrator.results_dir / f"{symbol}_{method}_reliability.png"
        calibrator.plot_reliability_diagram(predictions, y_true, str(plot_path))
        
        # Save report
        report = calibrator.get_calibration_report()
        report_path = calibrator.results_dir / f"{symbol}_{method}_report.csv"
        report.to_csv(report_path, index=False)
        logger.info(f"Calibration report saved to {report_path}")
    
    return calibrated, calibrator


if __name__ == '__main__':
    # Demo with synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic predictions (over-confident)
    y_true = np.random.binomial(1, 0.5, n_samples)
    
    # Model 1: Over-confident predictions
    raw_pred_1 = np.random.beta(2, 5, n_samples) * y_true + np.random.beta(5, 2, n_samples) * (1 - y_true)
    raw_pred_1 = raw_pred_1 * 0.3 + 0.35  # Compress to middle
    
    # Model 2: Under-confident predictions  
    raw_pred_2 = np.random.beta(5, 5, n_samples)
    
    predictions = {
        'classifier_1': raw_pred_1,
        'classifier_2': raw_pred_2,
    }
    
    # Calibrate
    calibrated, calibrator = calibrate_ensemble_predictions(
        predictions, y_true, method='isotonic', symbol='DEMO'
    )
    
    # Print report
    print("\nCalibration Report:")
    print(calibrator.get_calibration_report().to_string(index=False))
    print(f"\nCombination weights: {dict(zip(predictions.keys(), calibrator.weights_))}")

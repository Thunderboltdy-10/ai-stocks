"""
Standardized Training Logger for AI-Stocks

Creates structured log files that can be automatically analyzed by Claude.
All training scripts should use this logger for consistent output format.

Log files are saved to: python-ai-service/training_logs/{SYMBOL}_{MODEL}_{TIMESTAMP}.log

Usage:
    from utils.standardized_logger import TrainingLogger

    logger = TrainingLogger(symbol='AAPL', model_type='lstm')
    logger.log_config(epochs=50, batch_size=512, learning_rate=1e-3)
    logger.log_epoch(epoch=1, loss=0.02, val_loss=0.03, pred_std=0.05, pos_pct=0.48)
    logger.log_final_metrics(dir_acc=0.54, pred_std=0.042, wfe=62.3, sharpe=0.87)
    logger.log_status('SUCCESS')
    logger.close()
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


class TrainingLogger:
    """Standardized logger for training scripts.

    Creates structured log files with consistent format for automated analysis.
    """

    # Status constants
    STATUS_SUCCESS = 'SUCCESS'
    STATUS_FAILED = 'FAILED'
    STATUS_COLLAPSED = 'COLLAPSED'
    STATUS_BIAS = 'BIAS_DETECTED'
    STATUS_RUNNING = 'RUNNING'

    def __init__(
        self,
        symbol: str,
        model_type: str,
        log_dir: Optional[str] = None,
        also_print: bool = True
    ):
        """Initialize the training logger.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'SPY')
            model_type: Model type (e.g., 'lstm', 'gbm_xgb', 'gbm_lgb', 'xlstm', 'backtest')
            log_dir: Directory for logs. Defaults to python-ai-service/training_logs/
            also_print: Whether to also print to stdout
        """
        self.symbol = symbol.upper()
        self.model_type = model_type.lower()
        self.also_print = also_print
        self.start_time = datetime.now()
        self.warnings: List[str] = []
        self.epoch_metrics: List[Dict[str, Any]] = []

        # Setup log directory
        if log_dir is None:
            # Find the python-ai-service directory
            current_dir = Path(__file__).parent.parent  # utils -> python-ai-service
            log_dir = current_dir / 'training_logs'
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log filename with timestamp
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        self.log_filename = f"{self.symbol}_{self.model_type}_{timestamp}.log"
        self.log_path = log_dir / self.log_filename

        # Open log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')

        # Write header
        self._write_line(f"=== AI-STOCKS TRAINING LOG ===")
        self._write_line(f"Timestamp: {self.start_time.isoformat()}")
        self._write_line(f"Symbol: {self.symbol}")
        self._write_line(f"Model: {self.model_type}")
        self._write_line(f"Log File: {self.log_filename}")
        self._write_line("")

    def _write_line(self, line: str, flush: bool = True):
        """Write a line to log file and optionally to stdout."""
        self.log_file.write(line + '\n')
        if flush:
            self.log_file.flush()
        if self.also_print:
            print(line)

    def log_config(
        self,
        epochs: int = 0,
        batch_size: int = 0,
        learning_rate: float = 0.0,
        features: int = 0,
        sequence_length: int = 0,
        **kwargs
    ):
        """Log training configuration.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            features: Number of features
            sequence_length: Sequence length for LSTM
            **kwargs: Additional configuration parameters
        """
        self._write_line("=== TRAINING CONFIG ===")
        self._write_line(f"Symbol: {self.symbol}")
        self._write_line(f"Model: {self.model_type}")
        if epochs > 0:
            self._write_line(f"Epochs: {epochs}")
        if batch_size > 0:
            self._write_line(f"Batch Size: {batch_size}")
        if learning_rate > 0:
            self._write_line(f"Learning Rate: {learning_rate:.2e}")
        if features > 0:
            self._write_line(f"Features: {features}")
        if sequence_length > 0:
            self._write_line(f"Sequence Length: {sequence_length}")

        # Log additional kwargs
        for key, value in kwargs.items():
            self._write_line(f"{key.replace('_', ' ').title()}: {value}")

        self._write_line("")

    def log_data_info(
        self,
        train_samples: int = 0,
        val_samples: int = 0,
        test_samples: int = 0,
        target_mean: float = 0.0,
        target_std: float = 0.0,
        positive_pct: float = 0.0,
        **kwargs
    ):
        """Log data information.

        Args:
            train_samples: Number of training samples
            val_samples: Number of validation samples
            test_samples: Number of test samples
            target_mean: Mean of target variable
            target_std: Std of target variable
            positive_pct: Percentage of positive targets
        """
        self._write_line("=== DATA INFO ===")
        if train_samples > 0:
            self._write_line(f"Train Samples: {train_samples}")
        if val_samples > 0:
            self._write_line(f"Val Samples: {val_samples}")
        if test_samples > 0:
            self._write_line(f"Test Samples: {test_samples}")
        if target_std > 0:
            self._write_line(f"Target Mean: {target_mean:.6f}")
            self._write_line(f"Target Std: {target_std:.6f}")
        if positive_pct > 0:
            self._write_line(f"Positive Targets: {positive_pct:.1f}%")

        for key, value in kwargs.items():
            self._write_line(f"{key.replace('_', ' ').title()}: {value}")

        self._write_line("")

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        val_loss: Optional[float] = None,
        pred_std: Optional[float] = None,
        pos_pct: Optional[float] = None,
        dir_acc: Optional[float] = None,
        lr: Optional[float] = None,
        **kwargs
    ):
        """Log metrics for a single epoch.

        Args:
            epoch: Epoch number (1-indexed)
            loss: Training loss
            val_loss: Validation loss
            pred_std: Standard deviation of predictions
            pos_pct: Percentage of positive predictions (0-100)
            dir_acc: Directional accuracy (0-100)
            lr: Current learning rate
        """
        # Build epoch string
        parts = [f"Epoch {epoch}: loss={loss:.6f}"]

        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")
        if pred_std is not None:
            parts.append(f"pred_std={pred_std:.6f}")
        if pos_pct is not None:
            parts.append(f"pos_pct={pos_pct:.1f}%")
        if dir_acc is not None:
            parts.append(f"dir_acc={dir_acc:.1f}%")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")

        for key, value in kwargs.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")

        line = ", ".join(parts)
        self._write_line(line)

        # Store for analysis
        self.epoch_metrics.append({
            'epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'pred_std': pred_std,
            'pos_pct': pos_pct,
            'dir_acc': dir_acc,
            'lr': lr,
            **kwargs
        })

        # Check for warnings
        if pred_std is not None and pred_std < 0.005:
            self.add_warning(f"Epoch {epoch}: VARIANCE COLLAPSE RISK (pred_std={pred_std:.6f} < 0.005)")
        if pos_pct is not None and (pos_pct > 85 or pos_pct < 15):
            self.add_warning(f"Epoch {epoch}: PREDICTION BIAS ({pos_pct:.1f}% positive)")

    def log_gbm_iteration(
        self,
        iteration: int,
        train_metric: float,
        val_metric: Optional[float] = None,
        metric_name: str = 'rmse',
        **kwargs
    ):
        """Log GBM boosting iteration.

        Args:
            iteration: Boosting iteration number
            train_metric: Training metric value
            val_metric: Validation metric value
            metric_name: Name of the metric
        """
        parts = [f"Iter {iteration}: train_{metric_name}={train_metric:.6f}"]
        if val_metric is not None:
            parts.append(f"val_{metric_name}={val_metric:.6f}")

        for key, value in kwargs.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")

        self._write_line(", ".join(parts))

    def log_fold_result(
        self,
        fold: int,
        train_metric: float,
        val_metric: float,
        test_metric: Optional[float] = None,
        metric_name: str = 'dir_acc',
        **kwargs
    ):
        """Log walk-forward fold result.

        Args:
            fold: Fold number
            train_metric: Training metric
            val_metric: Validation metric
            test_metric: Test metric
            metric_name: Name of the metric
        """
        parts = [f"Fold {fold}: train_{metric_name}={train_metric:.4f}, val_{metric_name}={val_metric:.4f}"]
        if test_metric is not None:
            parts.append(f"test_{metric_name}={test_metric:.4f}")

        for key, value in kwargs.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")

        self._write_line(", ".join(parts))

    def log_final_metrics(
        self,
        dir_acc: Optional[float] = None,
        pred_std: Optional[float] = None,
        pos_pct: Optional[float] = None,
        wfe: Optional[float] = None,
        sharpe: Optional[float] = None,
        sortino: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        total_return: Optional[float] = None,
        buy_hold_return: Optional[float] = None,
        alpha: Optional[float] = None,
        win_rate: Optional[float] = None,
        trades: Optional[int] = None,
        rmse: Optional[float] = None,
        mae: Optional[float] = None,
        ic: Optional[float] = None,
        **kwargs
    ):
        """Log final training/validation metrics.

        Args:
            dir_acc: Directional accuracy (0-100)
            pred_std: Prediction standard deviation
            pos_pct: Positive prediction percentage (0-100)
            wfe: Walk-Forward Efficiency (0-100)
            sharpe: Sharpe ratio
            sortino: Sortino ratio
            max_drawdown: Maximum drawdown (0-100)
            total_return: Strategy total return (0-100 for percentage)
            buy_hold_return: Buy & hold return (0-100)
            alpha: Alpha vs benchmark
            win_rate: Win rate percentage
            trades: Number of trades
            rmse: Root mean squared error
            mae: Mean absolute error
            ic: Information coefficient
        """
        self._write_line("")
        self._write_line("=== FINAL METRICS ===")

        if dir_acc is not None:
            self._write_line(f"Direction Accuracy: {dir_acc:.2f}%")
        if pred_std is not None:
            self._write_line(f"Prediction Std: {pred_std:.6f}")
        if pos_pct is not None:
            self._write_line(f"Positive %: {pos_pct:.1f}%")
        if wfe is not None:
            self._write_line(f"WFE: {wfe:.1f}%")
        if sharpe is not None:
            self._write_line(f"Sharpe Ratio: {sharpe:.4f}")
        if sortino is not None:
            self._write_line(f"Sortino Ratio: {sortino:.4f}")
        if max_drawdown is not None:
            self._write_line(f"Max Drawdown: {max_drawdown:.2f}%")
        if total_return is not None:
            self._write_line(f"Strategy Return: {total_return:.2f}%")
        if buy_hold_return is not None:
            self._write_line(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        if alpha is not None:
            self._write_line(f"Alpha: {alpha:.4f}")
        if win_rate is not None:
            self._write_line(f"Win Rate: {win_rate:.1f}%")
        if trades is not None:
            self._write_line(f"Trades: {trades}")
        if rmse is not None:
            self._write_line(f"RMSE: {rmse:.6f}")
        if mae is not None:
            self._write_line(f"MAE: {mae:.6f}")
        if ic is not None:
            self._write_line(f"IC: {ic:.4f}")

        # Additional metrics
        for key, value in kwargs.items():
            if isinstance(value, float):
                self._write_line(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                self._write_line(f"{key.replace('_', ' ').title()}: {value}")

        self._write_line("")

    def add_warning(self, message: str):
        """Add a warning message.

        Args:
            message: Warning message
        """
        warning = f"[WARN] {message}"
        self.warnings.append(warning)
        self._write_line(warning)

    def add_error(self, message: str):
        """Add an error message.

        Args:
            message: Error message
        """
        error = f"[ERROR] {message}"
        self.warnings.append(error)
        self._write_line(error)

    def log_info(self, message: str):
        """Log an info message.

        Args:
            message: Info message
        """
        self._write_line(f"[INFO] {message}")

    def log_status(self, status: str, message: Optional[str] = None):
        """Log final status.

        Args:
            status: One of SUCCESS, FAILED, COLLAPSED, BIAS_DETECTED
            message: Optional additional message
        """
        self._write_line("")
        self._write_line("=== WARNINGS ===")
        if self.warnings:
            for warning in self.warnings:
                if not warning.startswith('['):
                    self._write_line(warning)
        else:
            self._write_line("None")

        self._write_line("")
        self._write_line("=== STATUS ===")
        self._write_line(status)
        if message:
            self._write_line(f"Message: {message}")

        # Log duration
        duration = datetime.now() - self.start_time
        self._write_line(f"Duration: {duration}")
        self._write_line(f"Log File: {self.log_path}")

    def close(self):
        """Close the log file."""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.add_error(f"{exc_type.__name__}: {exc_val}")
            self.log_status(self.STATUS_FAILED, str(exc_val))
        self.close()
        return False  # Don't suppress exceptions

    def get_log_path(self) -> str:
        """Get the full path to the log file."""
        return str(self.log_path)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the training run.

        Returns:
            Dictionary with key metrics
        """
        summary = {
            'symbol': self.symbol,
            'model': self.model_type,
            'log_file': self.log_filename,
            'start_time': self.start_time.isoformat(),
            'n_epochs': len(self.epoch_metrics),
            'n_warnings': len(self.warnings),
            'warnings': self.warnings
        }

        # Add final epoch metrics if available
        if self.epoch_metrics:
            last = self.epoch_metrics[-1]
            summary['final_loss'] = last.get('loss')
            summary['final_val_loss'] = last.get('val_loss')
            summary['final_pred_std'] = last.get('pred_std')
            summary['final_pos_pct'] = last.get('pos_pct')

        return summary


def get_latest_log(symbol: str, model_type: Optional[str] = None, log_dir: Optional[str] = None) -> Optional[str]:
    """Get the most recent log file for a symbol.

    Args:
        symbol: Stock symbol
        model_type: Optional model type filter
        log_dir: Log directory

    Returns:
        Path to latest log file, or None if not found
    """
    if log_dir is None:
        current_dir = Path(__file__).parent.parent
        log_dir = current_dir / 'training_logs'
    else:
        log_dir = Path(log_dir)

    if not log_dir.exists():
        return None

    pattern = f"{symbol.upper()}_"
    if model_type:
        pattern += f"{model_type.lower()}_"
    pattern += "*.log"

    logs = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    return str(logs[0]) if logs else None


def parse_log_file(log_path: str) -> Dict[str, Any]:
    """Parse a training log file and extract metrics.

    Args:
        log_path: Path to log file

    Returns:
        Dictionary with parsed metrics
    """
    result = {
        'file': log_path,
        'config': {},
        'data_info': {},
        'epoch_metrics': [],
        'final_metrics': {},
        'warnings': [],
        'status': None,
        'duration': None
    }

    current_section = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Detect sections
            if line.startswith('=== TRAINING CONFIG ==='):
                current_section = 'config'
                continue
            elif line.startswith('=== DATA INFO ==='):
                current_section = 'data_info'
                continue
            elif line.startswith('=== FINAL METRICS ==='):
                current_section = 'final_metrics'
                continue
            elif line.startswith('=== WARNINGS ==='):
                current_section = 'warnings'
                continue
            elif line.startswith('=== STATUS ==='):
                current_section = 'status'
                continue
            elif line.startswith('==='):
                current_section = None
                continue

            # Parse based on section
            if current_section == 'config' or current_section == 'data_info' or current_section == 'final_metrics':
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()

                    # Try to parse numeric values
                    try:
                        if '%' in value:
                            value = float(value.replace('%', ''))
                        elif '.' in value or 'e' in value.lower():
                            value = float(value)
                        elif value.isdigit():
                            value = int(value)
                    except ValueError:
                        pass

                    if current_section == 'config':
                        result['config'][key] = value
                    elif current_section == 'data_info':
                        result['data_info'][key] = value
                    else:
                        result['final_metrics'][key] = value

            elif current_section == 'warnings':
                if line != 'None':
                    result['warnings'].append(line)

            elif current_section == 'status':
                if line in ['SUCCESS', 'FAILED', 'COLLAPSED', 'BIAS_DETECTED', 'RUNNING']:
                    result['status'] = line
                elif line.startswith('Duration:'):
                    result['duration'] = line.replace('Duration:', '').strip()

            # Parse epoch metrics
            if line.startswith('Epoch '):
                epoch_data = {}
                parts = line.replace('Epoch ', '').split(', ')
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=', 1)
                        try:
                            if '%' in v:
                                v = float(v.replace('%', ''))
                            else:
                                v = float(v)
                        except ValueError:
                            pass
                        epoch_data[k.strip()] = v
                    elif ':' in part:
                        # First part like "1: loss=0.02"
                        epoch_num, rest = part.split(':', 1)
                        epoch_data['epoch'] = int(epoch_num.strip())
                        if '=' in rest:
                            k, v = rest.strip().split('=', 1)
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                            epoch_data[k.strip()] = v

                result['epoch_metrics'].append(epoch_data)

    return result


# Convenience functions for quick analysis
def check_variance_collapse(log_path: str, threshold: float = 0.005) -> bool:
    """Check if a training run had variance collapse.

    Args:
        log_path: Path to log file
        threshold: Variance threshold

    Returns:
        True if variance collapse detected
    """
    parsed = parse_log_file(log_path)

    # Check final metrics
    pred_std = parsed['final_metrics'].get('prediction_std')
    if pred_std is not None and pred_std < threshold:
        return True

    # Check epoch metrics
    for epoch in parsed['epoch_metrics']:
        if 'pred_std' in epoch and epoch['pred_std'] < threshold:
            return True

    return parsed['status'] == 'COLLAPSED'


def check_bias(log_path: str, threshold: float = 75.0) -> bool:
    """Check if a training run had prediction bias.

    Args:
        log_path: Path to log file
        threshold: Positive percentage threshold (either >threshold or <(100-threshold))

    Returns:
        True if bias detected
    """
    parsed = parse_log_file(log_path)

    pos_pct = parsed['final_metrics'].get('positive_%')
    if pos_pct is not None:
        if pos_pct > threshold or pos_pct < (100 - threshold):
            return True

    return parsed['status'] == 'BIAS_DETECTED'

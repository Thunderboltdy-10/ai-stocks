"""
Training Logger Utility for AI-Stocks.

Provides centralized logging configuration for training scripts with
both console and file output. Logs are saved to training_logs/ directory.

Usage:
    from utils.training_logger import setup_training_logger
    
    logger = setup_training_logger('AAPL', 'regressor')
    logger.info("Training started...")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_training_logger(
    symbol: str,
    model_type: str = 'regressor',
    log_dir: str = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Configure comprehensive logging for training scripts.
    
    Creates a logger with both console and file handlers:
    - Console: Shows INFO+ messages with simplified format
    - File: Logs DEBUG+ messages with full timestamps and context
    
    Args:
        symbol: Stock symbol (used in log filename)
        model_type: Type of model being trained ('regressor', 'classifiers', 'tft')
        log_dir: Directory for log files (default: training_logs/)
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
    
    Returns:
        Configured logger instance
    """
    # Determine log directory
    if log_dir is None:
        base_dir = Path(__file__).parent.parent  # python-ai-service/
        log_dir = base_dir / 'training_logs'
    else:
        log_dir = Path(log_dir)
    
    # Create log directory if needed
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{symbol}_{model_type}_{timestamp}.log"
    log_path = log_dir / log_filename
    
    # Create logger
    logger_name = f"training.{symbol}.{model_type}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with simplified format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler with detailed format
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Training log initialized: {log_path}")
    logger.debug(f"Symbol: {symbol}, Model Type: {model_type}")
    logger.debug(f"Python version: {sys.version}")
    
    return logger


def get_latest_log(symbol: str, model_type: str = 'regressor', log_dir: str = None) -> Path:
    """
    Get the path to the most recent log file for a symbol/model type.
    
    Args:
        symbol: Stock symbol
        model_type: Type of model
        log_dir: Directory containing logs (default: training_logs/)
    
    Returns:
        Path to the most recent log file, or None if not found
    """
    if log_dir is None:
        base_dir = Path(__file__).parent.parent
        log_dir = base_dir / 'training_logs'
    else:
        log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return None
    
    pattern = f"{symbol}_{model_type}_*.log"
    log_files = sorted(log_dir.glob(pattern), reverse=True)
    
    return log_files[0] if log_files else None


def cleanup_old_logs(
    symbol: str = None,
    model_type: str = None,
    log_dir: str = None,
    keep_count: int = 5
) -> int:
    """
    Clean up old log files, keeping only the most recent ones.
    
    Args:
        symbol: Stock symbol (None = all symbols)
        model_type: Model type (None = all types)
        log_dir: Directory containing logs
        keep_count: Number of recent logs to keep per symbol/type combo
    
    Returns:
        Number of log files deleted
    """
    if log_dir is None:
        base_dir = Path(__file__).parent.parent
        log_dir = base_dir / 'training_logs'
    else:
        log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return 0
    
    # Build pattern
    sym_pattern = symbol if symbol else '*'
    type_pattern = model_type if model_type else '*'
    pattern = f"{sym_pattern}_{type_pattern}_*.log"
    
    # Group files by symbol+type
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for log_file in log_dir.glob(pattern):
        parts = log_file.stem.rsplit('_', 1)  # Split off timestamp
        if len(parts) >= 2:
            key = parts[0]  # symbol_type
            grouped[key].append(log_file)
    
    # Delete old files, keep most recent
    deleted = 0
    for key, files in grouped.items():
        sorted_files = sorted(files, reverse=True)  # Newest first
        for old_file in sorted_files[keep_count:]:
            try:
                old_file.unlink()
                deleted += 1
            except Exception:
                pass
    
    return deleted

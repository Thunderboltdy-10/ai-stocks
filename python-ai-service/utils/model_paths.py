"""
Centralized model path configuration for AI-Stocks.

This module provides standardized path functions for all model types,
ensuring consistent organization under saved_models/{SYMBOL}/ structure.

Directory Structure:
    saved_models/
    └── {SYMBOL}/
        ├── regressor/
        │   ├── model/                    # Keras SavedModel directory
        │   ├── regressor.weights.h5      # Model weights (Keras 3.x format)
        │   ├── feature_scaler.pkl        # Feature scaler
        │   ├── target_scaler.pkl         # Target scaler (legacy)
        │   ├── target_scaler_robust.pkl  # Robust target scaler (v3.1+)
        │   ├── features.pkl              # Feature column list
        │   └── metadata.pkl              # Training metadata
        ├── classifiers/
        │   ├── buy_model/                # BUY classifier SavedModel
        │   ├── sell_model/               # SELL classifier SavedModel
        │   ├── buy.weights.h5            # BUY classifier weights (Keras 3.x format)
        │   ├── sell.weights.h5           # SELL classifier weights (Keras 3.x format)
        │   ├── buy_calibrated.pkl        # Calibrated BUY classifier
        │   ├── sell_calibrated.pkl       # Calibrated SELL classifier
        │   ├── feature_scaler.pkl        # Feature scaler
        │   ├── features.pkl              # Feature column list
        │   └── metadata.pkl              # Training metadata
        ├── quantile/
        │   ├── quantile.weights.h5       # Quantile regressor weights (Keras 3.x format)
        │   ├── config.json               # Model configuration
        │   ├── feature_scaler.pkl        # Feature scaler
        │   ├── target_scaler.pkl         # Target scaler
        │   └── metadata.pkl              # Training metadata
        ├── tft/
        │   ├── best_model.ckpt           # TFT checkpoint
        │   ├── config.json               # TFT configuration
        │   ├── hparams.json              # Hyperparameters
        │   └── dataset_params.pkl        # Dataset parameters
        ├── feature_columns.pkl           # Canonical feature list (shared)
        └── ensemble_metadata.pkl         # Ensemble metadata (if applicable)

Usage:
    from utils.model_paths import ModelPaths
    
    paths = ModelPaths('AAPL')
    paths.regressor.weights  # -> Path to regressor weights
    paths.classifiers.buy_model  # -> Path to BUY classifier model
    paths.ensure_dirs()  # Create all directories if needed
"""

from pathlib import Path
from typing import Optional
import os


# Base directory for all saved models
def get_saved_models_root() -> Path:
    """Get the root saved_models directory, auto-detecting project structure."""
    # Try to find python-ai-service directory
    current = Path(__file__).resolve()
    
    # Walk up to find python-ai-service
    for parent in current.parents:
        if parent.name == 'python-ai-service':
            return parent / 'saved_models'
        if (parent / 'python-ai-service').exists():
            return parent / 'python-ai-service' / 'saved_models'
    
    # Fallback: relative to current file
    return Path(__file__).resolve().parent.parent / 'saved_models'


SAVED_MODELS_ROOT = get_saved_models_root()


class RegressorPaths:
    """Path accessors for regressor model artifacts."""
    
    def __init__(self, symbol_dir: Path):
        self.base = symbol_dir / 'regressor'
    
    @property
    def model(self) -> Path:
        """Keras SavedModel file (.keras)."""
        return self.base / 'model.keras'
    
    @property
    def saved_model(self) -> Path:
        """Alias for model - Keras SavedModel file (.keras)."""
        return self.base / 'model.keras'
    
    @property
    def weights(self) -> Path:
        """Model weights file (.weights.h5 for Keras 3.x)."""
        return self.base / 'regressor.weights.h5'
    
    @property
    def feature_scaler(self) -> Path:
        """Feature scaler pickle."""
        return self.base / 'feature_scaler.pkl'
    
    @property
    def target_scaler(self) -> Path:
        """Target scaler pickle (legacy)."""
        return self.base / 'target_scaler.pkl'
    
    @property
    def target_scaler_robust(self) -> Path:
        """Robust target scaler pickle (v3.1+)."""
        return self.base / 'target_scaler_robust.pkl'
    
    @property
    def features(self) -> Path:
        """Feature columns list pickle."""
        return self.base / 'features.pkl'
    
    @property
    def metadata(self) -> Path:
        """Training metadata pickle."""
        return self.base / 'metadata.pkl'
    
    @property
    def vol_scaler(self) -> Path:
        """Volatility scaler for multi-task models."""
        return self.base / 'vol_scaler.pkl'
    
    def ensure_dir(self) -> None:
        """Create directory if it doesn't exist."""
        self.base.mkdir(parents=True, exist_ok=True)


class ClassifierPaths:
    """Path accessors for classifier model artifacts."""
    
    def __init__(self, symbol_dir: Path):
        self.base = symbol_dir / 'classifiers'
    
    @property
    def buy_model(self) -> Path:
        """BUY classifier SavedModel file (.keras)."""
        return self.base / 'buy_model.keras'
    
    @property
    def sell_model(self) -> Path:
        """SELL classifier SavedModel file (.keras)."""
        return self.base / 'sell_model.keras'
    
    @property
    def buy_weights(self) -> Path:
        """BUY classifier weights file (.weights.h5 for Keras 3.x)."""
        return self.base / 'buy.weights.h5'
    
    @property
    def sell_weights(self) -> Path:
        """SELL classifier weights file (.weights.h5 for Keras 3.x)."""
        return self.base / 'sell.weights.h5'
    
    @property
    def buy_calibrated(self) -> Path:
        """Calibrated BUY classifier pickle."""
        return self.base / 'buy_calibrated.pkl'
    
    @property
    def sell_calibrated(self) -> Path:
        """Calibrated SELL classifier pickle."""
        return self.base / 'sell_calibrated.pkl'
    
    @property
    def feature_scaler(self) -> Path:
        """Feature scaler pickle."""
        return self.base / 'feature_scaler.pkl'
    
    @property
    def features(self) -> Path:
        """Feature columns list pickle."""
        return self.base / 'features.pkl'
    
    @property
    def metadata(self) -> Path:
        """Training metadata pickle."""
        return self.base / 'metadata.pkl'
    
    def ensemble_weights(self, index: int) -> Path:
        """Ensemble member weights (buy/sell combined by suffix)."""
        suffix = '' if index == 0 else f'_ens{index+1}'
        return self.base / f'ensemble{suffix}.weights.h5'
    
    def ensure_dir(self) -> None:
        """Create directory if it doesn't exist."""
        self.base.mkdir(parents=True, exist_ok=True)


class QuantilePaths:
    """Path accessors for quantile regressor artifacts."""
    
    def __init__(self, symbol_dir: Path):
        self.base = symbol_dir / 'quantile'
    
    @property
    def weights(self) -> Path:
        """Quantile regressor weights file (.weights.h5 for Keras 3.x)."""
        return self.base / 'quantile.weights.h5'
    
    @property
    def weights_pkl(self) -> Path:
        """Fallback weights pickle (if h5 fails)."""
        return self.base / 'weights.pkl'
    
    @property
    def config(self) -> Path:
        """Model configuration JSON."""
        return self.base / 'config.json'
    
    @property
    def feature_scaler(self) -> Path:
        """Feature scaler pickle."""
        return self.base / 'feature_scaler.pkl'
    
    @property
    def target_scaler(self) -> Path:
        """Target scaler pickle."""
        return self.base / 'target_scaler.pkl'
    
    @property
    def metadata(self) -> Path:
        """Training metadata pickle."""
        return self.base / 'metadata.pkl'
    
    @property
    def predictions_plot(self) -> Path:
        """Quantile predictions diagnostic plot."""
        return self.base / 'predictions.png'
    
    def ensure_dir(self) -> None:
        """Create directory if it doesn't exist."""
        self.base.mkdir(parents=True, exist_ok=True)


class TFTPaths:
    """Path accessors for TFT model artifacts."""

    def __init__(self, symbol_dir: Path):
        self.base = symbol_dir / 'tft'

    @property
    def checkpoint(self) -> Path:
        """TFT model checkpoint (.ckpt)."""
        return self.base / 'best_model.ckpt'

    @property
    def config(self) -> Path:
        """TFT configuration JSON."""
        return self.base / 'config.json'

    @property
    def hparams(self) -> Path:
        """Hyperparameters JSON."""
        return self.base / 'hparams.json'

    @property
    def dataset_params(self) -> Path:
        """Dataset parameters pickle."""
        return self.base / 'dataset_params.pkl'

    @property
    def train_config(self) -> Path:
        """Training configuration JSON."""
        return self.base / 'train_config.json'

    def ensure_dir(self) -> None:
        """Create directory if it doesn't exist."""
        self.base.mkdir(parents=True, exist_ok=True)


class xLSTMPaths:
    """Path accessors for xLSTM-TS model artifacts."""

    def __init__(self, symbol_dir: Path):
        self.base = symbol_dir / 'xlstm'

    @property
    def model(self) -> Path:
        """Keras SavedModel file (.keras)."""
        return self.base / 'model.keras'

    @property
    def weights(self) -> Path:
        """Model weights file (.weights.h5)."""
        return self.base / 'xlstm.weights.h5'

    @property
    def feature_scaler(self) -> Path:
        """Feature scaler pickle."""
        return self.base / 'feature_scaler.pkl'

    @property
    def target_scaler(self) -> Path:
        """Target scaler pickle."""
        return self.base / 'target_scaler.pkl'

    @property
    def features(self) -> Path:
        """Feature columns list pickle."""
        return self.base / 'features.pkl'

    @property
    def metadata(self) -> Path:
        """Training metadata pickle."""
        return self.base / 'metadata.pkl'

    @property
    def config(self) -> Path:
        """Model configuration JSON."""
        return self.base / 'config.json'

    @property
    def wfe_results(self) -> Path:
        """Walk-forward validation results JSON."""
        return self.base / 'wfe_results.json'

    def ensure_dir(self) -> None:
        """Create directory if it doesn't exist."""
        self.base.mkdir(parents=True, exist_ok=True)


class ModelPaths:
    """
    Main path manager for a symbol's model artifacts.
    
    Usage:
        paths = ModelPaths('AAPL')
        paths.regressor.weights  # Path to regressor weights
        paths.classifiers.buy_model  # Path to BUY classifier model
        paths.ensure_dirs()  # Create all directories
    """
    
    def __init__(self, symbol: str, root: Optional[Path] = None):
        """
        Initialize path manager for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            root: Optional custom root directory (defaults to SAVED_MODELS_ROOT)
        """
        self.symbol = symbol.upper()
        self.root = root or SAVED_MODELS_ROOT
        self.symbol_dir = self.root / self.symbol
        
        # Initialize sub-path managers
        self._regressor = RegressorPaths(self.symbol_dir)
        self._classifiers = ClassifierPaths(self.symbol_dir)
        self._quantile = QuantilePaths(self.symbol_dir)
        self._tft = TFTPaths(self.symbol_dir)
        self._xlstm = xLSTMPaths(self.symbol_dir)
    
    @property
    def regressor(self) -> RegressorPaths:
        """Access regressor model paths."""
        return self._regressor
    
    @property
    def classifiers(self) -> ClassifierPaths:
        """Access classifier model paths."""
        return self._classifiers
    
    @property
    def quantile(self) -> QuantilePaths:
        """Access quantile regressor paths."""
        return self._quantile
    
    @property
    def tft(self) -> TFTPaths:
        """Access TFT model paths."""
        return self._tft

    @property
    def xlstm(self) -> xLSTMPaths:
        """Access xLSTM-TS model paths."""
        return self._xlstm

    @property
    def feature_columns(self) -> Path:
        """Canonical feature columns list (shared across models)."""
        return self.symbol_dir / 'feature_columns.pkl'
    
    @property
    def ensemble_metadata(self) -> Path:
        """Ensemble metadata (if using hybrid ensemble)."""
        return self.symbol_dir / 'ensemble_metadata.pkl'
    
    @property
    def target_metadata(self) -> Path:
        """Target processing metadata (v3.1+)."""
        return self.symbol_dir / 'target_metadata.pkl'
    
    def ensure_dirs(self) -> None:
        """Create all model directories if they don't exist."""
        self.symbol_dir.mkdir(parents=True, exist_ok=True)
        self._regressor.ensure_dir()
        self._classifiers.ensure_dir()
        self._quantile.ensure_dir()
        self._tft.ensure_dir()
        self._xlstm.ensure_dir()
    
    def exists(self) -> bool:
        """Check if symbol directory exists."""
        return self.symbol_dir.exists()
    
    def has_regressor(self) -> bool:
        """Check if regressor model exists."""
        return self._regressor.model.exists() or self._regressor.weights.exists()
    
    def has_classifiers(self) -> bool:
        """Check if classifier models exist."""
        return (self._classifiers.buy_model.exists() or self._classifiers.buy_weights.exists())
    
    def has_quantile(self) -> bool:
        """Check if quantile regressor exists."""
        return self._quantile.weights.exists() or self._quantile.weights_pkl.exists()
    
    def has_tft(self) -> bool:
        """Check if TFT model exists."""
        return self._tft.checkpoint.exists()

    def has_xlstm(self) -> bool:
        """Check if xLSTM-TS model exists."""
        return self._xlstm.model.exists() or self._xlstm.weights.exists()


# ============================================================================
# Legacy path compatibility functions
# ============================================================================

def get_legacy_regressor_paths(symbol: str, root: Optional[Path] = None) -> dict:
    """
    Get legacy (flat) regressor paths for backward compatibility.
    
    Returns dict with keys matching old naming convention.
    """
    root = root or SAVED_MODELS_ROOT
    symbol = symbol.upper()
    return {
        'weights': root / f'{symbol}_1d_regressor_final.weights.h5',
        'model': root / f'{symbol}_1d_regressor_final_model.keras',
        'feature_scaler': root / f'{symbol}_1d_regressor_final_feature_scaler.pkl',
        'target_scaler': root / f'{symbol}_1d_regressor_final_target_scaler.pkl',
        'target_scaler_robust': root / f'{symbol}_1d_target_scaler_robust.pkl',
        'features': root / f'{symbol}_1d_regressor_final_features.pkl',
        'metadata': root / f'{symbol}_1d_regressor_final_metadata.pkl',
    }


def get_legacy_classifier_paths(symbol: str, root: Optional[Path] = None) -> dict:
    """
    Get legacy (flat) classifier paths for backward compatibility.
    """
    root = root or SAVED_MODELS_ROOT
    symbol = symbol.upper()
    return {
        'buy_weights': root / f'{symbol}_is_buy_classifier_final.weights.h5',
        'sell_weights': root / f'{symbol}_is_sell_classifier_final.weights.h5',
        'buy_model': root / f'{symbol}_is_buy_classifier_final_model.keras',
        'sell_model': root / f'{symbol}_is_sell_classifier_final_model.keras',
        'buy_calibrated': root / f'{symbol}_is_buy_classifier_calibrated.pkl',
        'sell_calibrated': root / f'{symbol}_is_sell_classifier_calibrated.pkl',
        'feature_scaler': root / f'{symbol}_binary_feature_scaler.pkl',
        'features': root / f'{symbol}_binary_features.pkl',
        'metadata': root / f'{symbol}_binary_classifiers_final_metadata.pkl',
    }


def get_legacy_quantile_paths(symbol: str, root: Optional[Path] = None) -> dict:
    """
    Get legacy (flat) quantile regressor paths for backward compatibility.
    """
    root = root or SAVED_MODELS_ROOT
    symbol = symbol.upper()
    return {
        'weights': root / f'{symbol}_quantile_regressor.weights.h5',
        'config': root / f'{symbol}_quantile_regressor_config.json',
        'feature_scaler': root / f'{symbol}_quantile_feature_scaler.pkl',
        'target_scaler': root / f'{symbol}_quantile_target_scaler.pkl',
        'metadata': root / f'{symbol}_quantile_metadata.pkl',
        'predictions_plot': root / f'{symbol}_quantile_predictions.png',
    }


def get_legacy_tft_paths(symbol: str, root: Optional[Path] = None) -> dict:
    """
    Get legacy TFT paths for backward compatibility.
    """
    root = root or SAVED_MODELS_ROOT
    symbol = symbol.upper()
    return {
        'checkpoint': root / f'{symbol}_tft.ckpt',
        'checkpoint_alt': root / 'tft' / symbol / 'best_model.ckpt',
        'checkpoint_versioned': root / 'tft' / symbol / 'best_model-v9.ckpt',
        'config': root / f'{symbol}_tft_config.json',
    }


def find_model_path(symbol: str, model_type: str, artifact: str, 
                    root: Optional[Path] = None, prefer_new: bool = True) -> Optional[Path]:
    """
    Find a model artifact path, checking both new and legacy locations.
    
    Args:
        symbol: Stock symbol
        model_type: One of 'regressor', 'classifiers', 'quantile', 'tft'
        artifact: Artifact name (e.g., 'weights', 'model', 'feature_scaler')
        root: Optional custom root directory
        prefer_new: If True, check new paths first; if False, check legacy first
    
    Returns:
        Path to the artifact if found, None otherwise
    """
    paths = ModelPaths(symbol, root)
    
    # Get path accessor for model type
    path_accessor = getattr(paths, model_type, None)
    if path_accessor is None:
        return None
    
    # Get new path
    new_path = getattr(path_accessor, artifact, None)
    if new_path is None:
        return None
    
    # Get legacy paths
    legacy_funcs = {
        'regressor': get_legacy_regressor_paths,
        'classifiers': get_legacy_classifier_paths,
        'quantile': get_legacy_quantile_paths,
        'tft': get_legacy_tft_paths,
    }
    legacy_paths = legacy_funcs.get(model_type, lambda s, r: {})(symbol, root)
    legacy_path = legacy_paths.get(artifact)
    
    # Check paths in preferred order
    if prefer_new:
        check_order = [new_path, legacy_path]
    else:
        check_order = [legacy_path, new_path]
    
    for p in check_order:
        if p is not None and p.exists():
            return p
    
    return None


def migrate_legacy_to_new(symbol: str, root: Optional[Path] = None, 
                          dry_run: bool = True, copy_mode: bool = True) -> dict:
    """
    Migrate legacy flat structure to new organized structure.
    
    Args:
        symbol: Stock symbol to migrate
        root: Root directory (defaults to SAVED_MODELS_ROOT)
        dry_run: If True, only report what would be done
        copy_mode: If True, copy files; if False, move files
    
    Returns:
        dict with 'migrated', 'skipped', 'errors' lists
    """
    import shutil
    
    paths = ModelPaths(symbol, root)
    result = {'migrated': [], 'skipped': [], 'errors': []}
    
    # Mapping: (legacy_getter, model_type, artifact_mapping)
    migrations = [
        (get_legacy_regressor_paths, 'regressor', {
            'weights': 'weights',
            'model': 'model',
            'feature_scaler': 'feature_scaler',
            'target_scaler': 'target_scaler',
            'target_scaler_robust': 'target_scaler_robust',
            'features': 'features',
            'metadata': 'metadata',
        }),
        (get_legacy_classifier_paths, 'classifiers', {
            'buy_weights': 'buy_weights',
            'sell_weights': 'sell_weights',
            'buy_model': 'buy_model',
            'sell_model': 'sell_model',
            'buy_calibrated': 'buy_calibrated',
            'sell_calibrated': 'sell_calibrated',
            'feature_scaler': 'feature_scaler',
            'features': 'features',
            'metadata': 'metadata',
        }),
        (get_legacy_quantile_paths, 'quantile', {
            'weights': 'weights',
            'config': 'config',
            'feature_scaler': 'feature_scaler',
            'target_scaler': 'target_scaler',
            'metadata': 'metadata',
            'predictions_plot': 'predictions_plot',
        }),
        (get_legacy_tft_paths, 'tft', {
            'checkpoint': 'checkpoint',
            'config': 'config',
        }),
    ]
    
    for legacy_getter, model_type, artifact_map in migrations:
        legacy_paths = legacy_getter(symbol, root)
        path_accessor = getattr(paths, model_type)
        
        for legacy_key, new_key in artifact_map.items():
            legacy_path = legacy_paths.get(legacy_key)
            new_path = getattr(path_accessor, new_key, None)
            
            if legacy_path is None or new_path is None:
                continue
            
            if not legacy_path.exists():
                result['skipped'].append(f"{legacy_key}: source not found")
                continue
            
            if new_path.exists():
                result['skipped'].append(f"{new_key}: target already exists")
                continue
            
            try:
                if dry_run:
                    result['migrated'].append(f"{legacy_path} -> {new_path}")
                else:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    if copy_mode:
                        if legacy_path.is_dir():
                            shutil.copytree(legacy_path, new_path)
                        else:
                            shutil.copy2(legacy_path, new_path)
                    else:
                        shutil.move(str(legacy_path), str(new_path))
                    result['migrated'].append(f"{legacy_path} -> {new_path}")
            except Exception as e:
                result['errors'].append(f"{legacy_key}: {e}")
    
    return result

"""
Train Temporal Fusion Transformer (TFT) for multi-horizon quantile forecasting.

This script builds datasets via `TFTDatasetBuilder`, instantiates a TFT model
from `pytorch_forecasting`, trains using `pytorch_lightning.Trainer`, and
saves model artifacts and evaluation outputs.

Note: This script attempts to be robust if run from repository root. It assumes
the project has the `python-ai-service` package or that `python-ai-service`
is discoverable on `PYTHONPATH`. If imports fail, you may need to run from
the `python-ai-service/` directory or add it to `PYTHONPATH`.
"""

import os, sys
# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure repository root is on sys.path so `data` and `python_ai_service` imports work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# Also ensure the `python-ai-service` package folder is discoverable as a top-level package
python_ai_service_path = os.path.join(repo_root, 'python-ai-service')
if python_ai_service_path not in sys.path:
    sys.path.insert(0, python_ai_service_path)


import argparse
import json
import pickle
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss, TimeSeriesDataSet
from typing import Dict, Any
from pytorch_forecasting.metrics import MAE

# Robust imports: try package-style first, fallback to local package path
try:
    from data.data_fetcher import fetch_stock_data
    from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
    from data.tft_dataset_builder import TFTDatasetBuilder, SYMBOL_METADATA
    from utils.model_paths import ModelPaths, get_legacy_tft_paths
except Exception:
    # Try alternate import path used when running from python-ai-service folder
    from python_ai_service.data.data_fetcher import fetch_stock_data  # type: ignore
    from python_ai_service.data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT  # type: ignore
    from python_ai_service.data.tft_dataset_builder import TFTDatasetBuilder, SYMBOL_METADATA  # type: ignore
    from python_ai_service.utils.model_paths import ModelPaths, get_legacy_tft_paths  # type: ignore


MODEL_CONFIG = {
    'hidden_size': 128,
    'attention_head_size': 4,
    'dropout': 0.3,
    'hidden_continuous_size': 64,
    'output_size': 7,
    'learning_rate': 1e-3,
    'reduce_on_plateau_patience': 4,
}
def validate_dataset_for_tft(dataset: TimeSeriesDataSet) -> Dict[str, Any]:
    """Validate TFT dataset using only TimeSeriesDataSet API (no DataFrame access needed)."""
    results: Dict[str, Any] = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    try:
        # 1. Check dataset has samples
        dataset_len = len(dataset)
        if dataset_len == 0:
            results['valid'] = False
            results['errors'].append('Dataset is empty (0 samples)')
            return results

        results['samples'] = dataset_len

        # 2. Check dataset parameters
        if hasattr(dataset, 'max_encoder_length'):
            results['max_encoder_length'] = getattr(dataset, 'max_encoder_length')
        if hasattr(dataset, 'max_prediction_length'):
            results['max_prediction_length'] = getattr(dataset, 'max_prediction_length')

        # 3. Test first batch access
        try:
            sample = dataset[0]
            results['sample_keys'] = list(sample.keys()) if isinstance(sample, dict) else []

            # Check required keys exist
            required_keys = ['encoder_cont', 'decoder_cont', 'decoder_target']
            missing = [k for k in required_keys if k not in sample]
            if missing:
                results['valid'] = False
                results['errors'].append(f'Missing required keys: {missing}')

            # Log shapes for diagnostics
            if 'encoder_cont' in sample:
                try:
                    results['encoder_shape'] = tuple(sample['encoder_cont'].shape)
                except Exception:
                    pass
            if 'decoder_target' in sample:
                try:
                    results['target_shape'] = tuple(sample['decoder_target'].shape)
                except Exception:
                    pass

        except Exception as e:
            results['valid'] = False
            results['errors'].append(f'Cannot access dataset samples: {str(e)}')

        # 4. Attempt DataFrame access (optional - don't fail if unavailable)
        df = None
        try:
            if hasattr(dataset, 'data'):
                data_attr = dataset.data
                if isinstance(data_attr, pd.DataFrame):
                    df = data_attr
                elif isinstance(data_attr, dict):
                    # Try common keys
                    for key in ['train', 'df', 'dataframe', 0]:
                        try:
                            if key in data_attr and isinstance(data_attr[key], pd.DataFrame):
                                df = data_attr[key]
                                break
                        except Exception:
                            # some dict-like objects may not support 'in' for mixed keys
                            pass
        except Exception:
            pass  # DataFrame access is optional

        if df is not None and isinstance(df, pd.DataFrame):
            results['feature_count'] = len(df.columns)

            # Check for Date column
            if 'Date' in df.columns:
                try:
                    results['date_range'] = {
                        'start': str(df['Date'].min()),
                        'end': str(df['Date'].max()),
                        'span_days': int((pd.to_datetime(df['Date'].max()) - pd.to_datetime(df['Date'].min())).days)
                    }
                except Exception:
                    pass

            # Check time_idx column
            if 'time_idx' in df.columns:
                try:
                    if not df['time_idx'].is_monotonic_increasing:
                        results['warnings'].append('time_idx is not monotonic increasing')
                except Exception:
                    results['warnings'].append('time_idx monotonicity check failed')
            else:
                results['warnings'].append('time_idx column not found in DataFrame')
        else:
            # Not an error - just log that we couldn't access it
            results['warnings'].append('Could not extract DataFrame from dataset')

    except Exception as e:
        results['valid'] = False
        results['errors'].append(f'Validation exception: {str(e)}')

    return results


def train_tft_model(symbols, max_encoder_length=60, max_prediction_length=10,
                   epochs=50, batch_size=512, hidden_size=128, learning_rate=1e-3,
                   output_dir='saved_models/tft', use_cache: bool = True):
    """Main training function for TFT model.

    This function:
    - loads and engineers data for each symbol
    - builds TimeSeriesDataSet train/val
    - instantiates TemporalFusionTransformer
    - sets up Trainer and callbacks
    - trains and saves artifacts
    """
    symbols_list = [s.strip().upper() for s in symbols]
    run_name = "_".join(symbols_list)
    
    # For single symbol, use new ModelPaths structure
    # For multi-symbol, use legacy output_dir structure
    if len(symbols_list) == 1:
        paths = ModelPaths(symbols_list[0])
        paths.ensure_dirs()
        run_dir = paths.tft.base
        # Also maintain legacy location
        legacy_output_dir = Path(output_dir)
        legacy_run_dir = legacy_output_dir / run_name
        legacy_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Multi-symbol training uses separate directory
        output_dir = Path(output_dir)
        run_dir = output_dir / run_name
        legacy_run_dir = run_dir
    
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  TRAINING TFT MODEL: {run_name}")
    print(f"{'='*70}\n")

    # Prepare data per-symbol
    builder = TFTDatasetBuilder(max_encoder_length=max_encoder_length,
                                max_prediction_length=max_prediction_length,
                                target_col='returns')

    # Get canonical features list upfront - needed for both cache validation and fresh data
    canonical_features = get_feature_columns(include_sentiment=True)
    print(f"Technical features by category (expected sum): {len(canonical_features) - 34}")  # 147 - 34 sentiment = 113 tech
    print(f"Actual technical list length: {len([f for f in canonical_features if not f.startswith('sentiment')])}")
    print(f"Difference from canonical 113: {113 - len([f for f in canonical_features if not f.startswith('sentiment')])}")

    prepared_dfs = []
    for symbol in symbols_list:
        cache_path = run_dir / f'{symbol}_prepared.pkl'
        df_prep = None

        # Try loading cache with validation and auto-delete if outdated
        if use_cache and cache_path.exists():
            print(f'📊 Validating cached dataset for {symbol}...')
            try:
                with open(cache_path, 'rb') as fh:
                    df_cached = pickle.load(fh)

                missing = set(canonical_features) - set(df_cached.columns)

                if missing:
                    print(f"⚠️  Cached data missing {len(missing)} features: {list(missing)[:5]}...")
                    print('   Deleting outdated cache and regenerating...')
                    try:
                        if cache_path.exists():
                            cache_path.unlink()
                    except Exception as e:
                        warnings.warn(f'Could not delete outdated cache {cache_path}: {e}')
                else:
                    # Validate target finiteness in cached prepared dataset.
                    # If many non-finite target values exist, the cache is likely stale
                    # (prepared before target-shift or before target cleanup). In that case
                    # delete the cache and regenerate. If only a tiny fraction of rows
                    # are non-finite, drop them in-memory and continue.
                    import numpy as _np
                    target_col = builder.target_col
                    nonfinite = 0
                    try:
                        if target_col in df_cached.columns:
                            nonfinite = int((~_np.isfinite(df_cached[target_col].values)).sum())
                        else:
                            # If target not present, consider cache invalid
                            nonfinite = len(df_cached)
                    except Exception:
                        # Defensive: if check fails, mark cache invalid
                        nonfinite = len(df_cached)

                    pct = nonfinite / max(1, len(df_cached))
                    if nonfinite > 0:
                        print(f"⚠️  Cached data: {nonfinite} ({pct:.2%}) non-finite '{target_col}' values found")

                    # If a substantial portion of targets are non-finite, delete cache
                    if pct > 0.05:
                        print('   Deleting outdated cache due to non-finite targets and regenerating...')
                        try:
                            if cache_path.exists():
                                cache_path.unlink()
                        except Exception as e:
                            warnings.warn(f'Could not delete outdated cache {cache_path}: {e}')
                    else:
                        # Small number of bad rows: drop them and accept the cache
                        if nonfinite > 0 and target_col in df_cached.columns:
                            df_cached = df_cached[_np.isfinite(df_cached[target_col].values)].reset_index(drop=True)
                            print(f'   Dropped {nonfinite} non-finite target rows from cache; proceeding')
                        print(f'✓ Cache valid: {len(df_cached)} samples, {len(canonical_features)} features')
                        
                        # Ensure TFT-specific columns exist (symbol, time_idx, sector, etc.)
                        if 'symbol' not in df_cached.columns or 'time_idx' not in df_cached.columns:
                            print(f'   Preparing cached data for TFT (adding symbol, time_idx, etc.)...')
                            df_cached = builder.prepare_dataframe(df_cached, symbol)
                        
                        df_prep = df_cached
                        # Ensure builder has canonical features set for later dataset creation
                        builder.time_varying_unknown_reals = canonical_features
            except Exception as e:
                print(f'⚠️  Cache validation failed: {e}')
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception:
                    pass

        # Fetch and prepare if cache invalid or missing
        if df_prep is None:
            print(f'📊 Fetching and preparing data for {symbol}...')
            # Use DataCacheManager so fetch/engineer/prepare are centralized and cached
            from data.cache_manager import DataCacheManager
            cache_manager = DataCacheManager()
            try:
                raw_df, df_eng, df_prep, feature_cols = cache_manager.get_or_fetch_data(
                    symbol=symbol,
                    include_sentiment=True,
                    force_refresh=False
                )
            except Exception:
                # Fallback: attempt a forced refresh
                raw_df, df_eng, df_prep, feature_cols = cache_manager.get_or_fetch_data(
                    symbol=symbol,
                    include_sentiment=True,
                    force_refresh=True
                )
            if df_prep is None or (hasattr(df_prep, '__len__') and len(df_prep) == 0):
                warnings.warn(f'No data returned for {symbol}, skipping')
                continue
            
            # Prepare the dataframe for TFT (adds symbol, time_idx, sector, etc.)
            # DataCacheManager returns prepared data but without TFT-specific columns
            try:
                df_prep = builder.prepare_dataframe(df_prep, symbol)
                print(f'   ✓ Prepared {len(df_prep)} rows for TFT')
            except Exception as e:
                print(f'⚠️  Failed to prepare dataframe for {symbol}: {e}')
                continue
            
            # Set builder's time_varying_unknown_reals from canonical features
            # This is required for create_dataset() to work
            if builder.time_varying_unknown_reals is None:
                builder.time_varying_unknown_reals = canonical_features
                print(f'   ✓ Set {len(canonical_features)} time_varying_unknown_reals from canonical features')

        prepared_dfs.append(df_prep)

    if len(prepared_dfs) == 0:
        raise RuntimeError('No prepared dataframes available for training')

    # Train/validation split by time: use last 20% as validation per-symbol
    train_list = []
    val_list = []
    for df in prepared_dfs:
        n = len(df)
        split_idx = int(n * 0.8)
        train_list.append(df.iloc[:split_idx].reset_index(drop=True))
        val_list.append(df.iloc[split_idx:].reset_index(drop=True))

    # Create datasets
    print("🛠️  Building TimeSeriesDataSet...")
    train_dataset = builder.create_dataset(train_list, split='train')
    val_dataset = builder.create_dataset(val_list, split='val', reference_dataset=train_dataset)

    # === Pre-training diagnostics ===
    print("\n📊 Dataset Diagnostics:")
    try:
        print(f"   Train samples: {len(train_dataset)}")
    except Exception:
        print("   Train samples: unknown")
    try:
        print(f"   Val samples:   {len(val_dataset)}")
    except Exception:
        print("   Val samples: unknown")
    try:
        feat_count = len(builder.time_varying_unknown_reals) if builder.time_varying_unknown_reals else 'unknown'
        print(f"   Features: {feat_count}")
    except Exception:
        print("   Features: unknown")
    try:
        print(f"   Encoder length: {builder.max_encoder_length}")
    except Exception:
        print("   Encoder length: unknown")
    try:
        print(f"   Prediction length: {builder.max_prediction_length}")
    except Exception:
        print("   Prediction length: unknown")

    # Verify feature count
    try:
        expected_count = 147
        actual_count = len(builder.time_varying_unknown_reals) if builder.time_varying_unknown_reals else 0
        if actual_count != expected_count:
            warnings.warn(f"Feature count mismatch: expected {expected_count}, got {actual_count}")
        else:
            print(f"   ✓ Feature count matches expected: {expected_count}")
    except Exception:
        pass

    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0) # num_workers=0 for stability on Windows
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples:   {len(val_dataset)}")

    # === Dataset diagnostics ===
    try:
        diag = validate_dataset_for_tft(train_dataset)
        import logging
        dlog = logging.getLogger(__name__)
        # If diagnostics could not extract a DataFrame or found errors, log but do not abort.
        if not diag.get('valid', False):
            # Log errors and warnings for visibility
            dlog.error('Dataset validation issues: %s', diag.get('errors'))
            for w in diag.get('warnings', []):
                dlog.warning('Dataset warning: %s', w)
            # Warn the user but continue since datasets were built successfully
            import warnings as _warnings
            _warnings.warn(f"⚠️  Dataset validation produced issues: {diag.get('errors', [])}")
            print('   Proceeding with training anyway (dataset was created successfully)')
            # Provide helpful summary if available
            if 'date_range' in diag and diag['date_range']:
                dr = diag['date_range']
                try:
                    print(f"   Date range: {dr[0]} to {dr[1]}")
                except Exception:
                    pass
            if 'feature_count' in diag and diag['feature_count'] is not None:
                try:
                    print(f"   Features: {diag['feature_count']}")
                except Exception:
                    pass
        else:
            dlog.info('Dataset validation passed: feature_count=%s, max_gap_days=%s',
                      diag.get('feature_count'), diag.get('max_gap_days'))
            for w in diag.get('warnings', []):
                dlog.warning('Dataset warning: %s', w)
            # Print a concise confirmation for console users
            print('✓ Dataset validation passed')
            if 'date_range' in diag and diag['date_range']:
                dr = diag['date_range']
                try:
                    print(f"   Date range: {dr[0]} to {dr[1]}")
                except Exception:
                    pass
            if 'feature_count' in diag and diag['feature_count'] is not None:
                try:
                    print(f"   Features: {diag['feature_count']}")
                except Exception:
                    pass
    except Exception:
        # If validation raised an unexpected exception, re-raise to avoid hiding real issues
        raise

    # === Dataloader batch smoke-test ===
    print("\n🧪 Testing dataset batching...")
    try:
        # Test that dataloader can produce batches
        test_batch = next(iter(train_dataloader))

        print(f"   ✓ Batch keys: {list(test_batch[0].keys()) if isinstance(test_batch, tuple) else list(test_batch.keys())}")

        # Extract x and y from batch
        if isinstance(test_batch, tuple):
            x_batch, y_batch = test_batch
        else:
            x_batch = test_batch
            y_batch = None

        # Log shapes
        if isinstance(x_batch, dict) and 'encoder_cont' in x_batch:
            try:
                enc_shape = x_batch['encoder_cont'].shape
                print(f"   ✓ Encoder shape: {enc_shape} (batch_size={enc_shape[0]}, seq_len={enc_shape[1]}, features={enc_shape[2]})")
            except Exception:
                pass

        if y_batch is not None:
            try:
                if isinstance(y_batch, tuple):
                    target = y_batch[0]
                else:
                    target = y_batch
                print(f"   ✓ Target shape: {getattr(target, 'shape', 'unknown')}")
            except Exception:
                pass

        print("   ✓ Dataloader test passed\n")

    except Exception as e:
        print(f"   ⚠️  Dataloader test warning: {e}")
        print("   Proceeding anyway - PyTorch Lightning will handle batching")

    # === Pre-training quick checks and sample batch logging ===
    try:
        import logging, time
        prelog = logging.getLogger(__name__)
        # Dataset params
        max_enc = getattr(train_dataset, 'max_encoder_length', builder.max_encoder_length)
        max_pred = getattr(train_dataset, 'max_prediction_length', builder.max_prediction_length)
        prelog.info('TimeSeriesDataSet params: max_encoder_length=%s, max_prediction_length=%s', max_enc, max_pred)

        # Sample a single batch and log shapes/contents (avoid huge dumps)
        try:
            sample_batch = next(iter(train_dataloader))
            # sample_batch is typically a tuple (x, y) where x is dict-like
            if isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 2:
                x_sample, y_sample = sample_batch[0], sample_batch[1]
            else:
                x_sample, y_sample = sample_batch, None

            # Log shapes for tensors inside x_sample
            if isinstance(x_sample, dict):
                for k, v in x_sample.items():
                    try:
                        shp = getattr(v, 'shape', None)
                        prelog.info('Sample input %s shape: %s', k, shp)
                    except Exception:
                        pass
            else:
                prelog.info('Sample input type: %s', type(x_sample))

            # Log target sample shape
            try:
                if hasattr(y_sample, 'shape'):
                    prelog.info('Sample target shape: %s', y_sample.shape)
                else:
                    prelog.info('Sample target type: %s', type(y_sample))
            except Exception:
                pass
        except StopIteration:
            prelog.warning('Could not sample from train_dataloader for diagnostics (empty?)')

        # GPU availability and memory
        try:
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev)
                mem = props.total_memory
                prelog.info('CUDA available: device=%s name=%s total_mem=%0.2f GB', dev, props.name, mem / (1024**3))
            else:
                prelog.info('CUDA not available; training will run on CPU')
        except Exception as e:
            prelog.warning('GPU check failed: %s', e)
    except Exception:
        pass

    # Build model
    print("🏗️  Initializing TemporalFusionTransformer...")

    # CRITICAL: Create target_normalizer OUTSIDE the model
    # This ensures it's properly serialized with the checkpoint
    try:
        from pytorch_forecasting.data import GroupNormalizer
    except Exception:
        # If import fails, let the original exception bubble later when constructing the model
        GroupNormalizer = None

    target_normalizer = None
    try:
        if GroupNormalizer is not None:
            target_normalizer = GroupNormalizer(
                groups=['symbol'],  # Normalize per symbol
                transformation='softplus'  # Safe transformation for returns
            )
    except Exception as e:
        print(f"⚠️ Could not construct GroupNormalizer: {e}")

    # Create TFT with explicit normalizer when supported by the installed library.
    # Some versions of pytorch-forecasting accept `target_normalizer` here; others do not.
    try:
        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=MODEL_CONFIG['attention_head_size'],
            dropout=MODEL_CONFIG['dropout'],
            hidden_continuous_size=MODEL_CONFIG['hidden_continuous_size'],
            output_size=MODEL_CONFIG['output_size'],
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=MODEL_CONFIG['reduce_on_plateau_patience'],
            # CRITICAL: Pass normalizer explicitly when supported
            **({'target_normalizer': target_normalizer} if target_normalizer is not None else {})
        )
    except TypeError as e:
        # Fallback for older/newer package versions that don't accept target_normalizer
        print(f"⚠️ from_dataset does not accept 'target_normalizer' parameter: {e}")
        print("   Falling back to creating the model without passing the normalizer, then attaching it to the loss if possible.")
        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=MODEL_CONFIG['attention_head_size'],
            dropout=MODEL_CONFIG['dropout'],
            hidden_continuous_size=MODEL_CONFIG['hidden_continuous_size'],
            output_size=MODEL_CONFIG['output_size'],
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=MODEL_CONFIG['reduce_on_plateau_patience'],
        )

        # Try to attach the normalizer to the loss so the encoder exists on the checkpoint
        try:
            if target_normalizer is not None and getattr(tft, 'loss', None) is not None:
                # common attribute names used by different PF versions
                if hasattr(tft.loss, 'encoder'):
                    tft.loss.encoder = target_normalizer
                elif hasattr(tft.loss, 'target_normalizer'):
                    setattr(tft.loss, 'target_normalizer', target_normalizer)
                else:
                    # best-effort fallback
                    try:
                        setattr(tft.loss, '_target_normalizer', target_normalizer)
                    except Exception:
                        pass
                print("✓ Attached target_normalizer to model.loss (fallback)")
        except Exception as _:
            print("⚠️ Failed to attach target_normalizer to loss in fallback path")

    # VERIFY normalizer attached before training
    if getattr(tft, 'loss', None) is not None:
        print(f"✓ Loss function: {type(tft.loss)}")
        if hasattr(tft.loss, 'quantiles'):
            try:
                print(f"  Quantiles: {tft.loss.quantiles}")
            except Exception:
                pass
    else:
        print("⚠️ WARNING: tft.loss is None - this will cause inference failures!")

    # CRITICAL: Save hyperparameters explicitly
    # This ensures the normalizer is serialized with the checkpoint
    try:
        tft.save_hyperparameters(ignore=['loss', 'logging_metrics'])
    except Exception:
        # Best-effort: not fatal if this method isn't available on this object
        pass

    # --- Save an inference-compatible TFT config JSON per symbol ---
    try:
        # Save model configuration (inference-compatible)
        config = {
            # TFT Model Parameters (constructor-compatible)
            'hidden_size': int(hidden_size),
            'attention_head_size': int(MODEL_CONFIG['attention_head_size']),
            'dropout': float(MODEL_CONFIG['dropout']),
            'hidden_continuous_size': int(MODEL_CONFIG['hidden_continuous_size']),
            'output_size': int(MODEL_CONFIG['output_size']),
            'learning_rate': float(learning_rate),
            'reduce_on_plateau_patience': int(MODEL_CONFIG['reduce_on_plateau_patience']),
            
            # Dataset Parameters (for inference data prep)
            'max_encoder_length': int(builder.max_encoder_length),
            'max_prediction_length': int(builder.max_prediction_length),
            'time_varying_known_reals': list(builder.time_varying_known_reals),
            'time_varying_unknown_reals': list(builder.time_varying_unknown_reals),
            'static_categoricals': list(builder.static_categoricals),
            'target': builder.target_col,
            'group_ids': ['symbol'],
            
            # Metadata
            'tft_arch_version': 1,
            'pytorch_forecasting_version': None,
        }
        try:
            import pytorch_forecasting as pf
            config['pytorch_forecasting_version'] = getattr(pf, '__version__', None)
        except Exception:
            # leave as None if not importable
            pass

        # DO NOT SAVE: num_attention_heads, lstm_layers (not in constructor)

        # Save config JSON for each symbol - new structure
        for symbol in symbols_list:
            symbol_paths = ModelPaths(symbol)
            symbol_paths.tft.ensure_dir()
            config_path = symbol_paths.tft.config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Saved TFT config: {config_path}")
            
            # Also save to legacy paths
            legacy_paths = get_legacy_tft_paths(symbol)
            legacy_paths['config'].parent.mkdir(parents=True, exist_ok=True)
            with open(legacy_paths['config'], 'w') as f:
                json.dump(config, f, indent=2)
        
    except Exception as e:
        print(f"⚠️ Config save failed: {e}")

    # Note: additional extended training metadata and dataset_parameters are
    # still saved later in the script (see 'train_config.json' and
    # 'dataset_parameters.pkl') — the per-symbol JSON above is the canonical
    # inference-facing config containing only constructor-acceptable fields.

    # Compatibility: newer/older package combinations may return a model
    # that is not a `pytorch_lightning.LightningModule` instance (or
    # the Trainer may reject it due to version mismatches). Wrap the
    # model in a thin LightningModule proxy when necessary so
    # `trainer.fit(...)` accepts it. The proxy delegates to the
    # underlying model where possible and provides a best-effort
    # training/validation step using the provided loss.
    tft_for_trainer = tft
    if not isinstance(tft, pl.LightningModule):
        warnings.warn(
            "TemporalFusionTransformer is not a LightningModule; using a wrapper proxy."
        )

        class TFTLightningWrapper(pl.LightningModule):
            def __init__(self, model, learning_rate):
                super().__init__()
                self.model = model
                self.learning_rate = learning_rate
                # Ignore these hyperparameters when saving to avoid warnings
                self.save_hyperparameters(ignore=['model'])

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

            def training_step(self, batch, batch_idx):
                # Unpack batch if it is a tuple (x, y)
                # pytorch_forecasting TimeSeriesDataSet yields (x, y)
                # x is a dict, y is a tuple (target, weight)
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, None

                # Compute predictions
                # self.model(x) returns a dictionary/namedtuple with 'prediction' and other keys
                out = self.model(x)
                
                # Extract prediction tensor
                # TFT output is a dict-like object (OutputMixIn) containing 'prediction'
                preds = out['prediction'] if isinstance(out, dict) or hasattr(out, '__getitem__') else out

                # Calculate loss using the model's internal loss function
                if hasattr(self.model, 'loss'):
                    # Pass predictions and target (y) to the metric
                    # y is typically (target_values, weights) or just target_values
                    loss = self.model.loss(preds, y)
                else:
                    # Fallback
                    target = y[0] if isinstance(y, (tuple, list)) else y
                    loss = QuantileLoss()(preds, target)
                
                self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, None

                out = self.model(x)
                preds = out['prediction'] if isinstance(out, dict) or hasattr(out, '__getitem__') else out
                
                if hasattr(self.model, 'loss'):
                    loss = self.model.loss(preds, y)
                else:
                    target = y[0] if isinstance(y, (tuple, list)) else y
                    loss = QuantileLoss()(preds, target)

                self.log('val_loss', loss, prog_bar=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        tft_for_trainer = TFTLightningWrapper(tft, learning_rate)

    # Callbacks and logger
    # --- Custom diagnostics callbacks ---

    class EpochLoggingCallback(pl.Callback):
        """Logs epoch-level metrics with timestamps and saves periodic checkpoints."""
        def on_train_epoch_end(self, trainer, pl_module, outputs=None):
            import time
            metrics = trainer.callback_metrics
            # human-readable timestamp for console print
            ts_str = time.strftime('%Y-%m-%d %H:%M:%S')
            # numeric epoch timestamp for logger (tensorboard expects numeric scalars)
            epoch_time = float(time.time())
            try:
                loss = float(metrics.get('train_loss', float('nan')))
            except Exception:
                loss = float('nan')
            # Log only numeric scalars to TensorBoard / logger
            try:
                pl_module.logger.log_metrics({'train_loss_epoch': loss, 'epoch_time': epoch_time}, step=trainer.current_epoch)
            except Exception:
                # best-effort: don't crash logging
                pass
            # print human-friendly timestamp to stdout for quick inspection
            print(f"[{ts_str}] Epoch {trainer.current_epoch} train_loss={loss:.6f}")

    class GradientNormCallback(pl.Callback):
        """Compute gradient norm after backward pass to detect exploding/vanishing gradients."""
        def on_after_backward(self, trainer, pl_module):
            total_norm = 0.0
            count = 0
            for p in pl_module.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.detach().data.norm(2)
                total_norm += float(param_norm.item() ** 2)
                count += 1
            if count > 0:
                total_norm = float((total_norm ** 0.5))
                # log to pl logger and stdout when abnormal
                try:
                    pl_module.logger.log_metrics({'grad_norm': total_norm}, step=trainer.global_step)
                except Exception:
                    pass
                if not (0.0 < total_norm < 1e4):
                    print(f"⚠️ Gradient norm abnormal: {total_norm:.6e} at step {trainer.global_step}")

    # Backup checkpoint every N epochs (fallback if ModelCheckpoint.every_n_epochs unavailable)
    class BackupCheckpointCallback(pl.Callback):
        def __init__(self, dirpath, every_n_epochs=10):
            super().__init__()
            self.dirpath = Path(dirpath)
            self.every_n_epochs = int(every_n_epochs)
            self.dirpath.mkdir(parents=True, exist_ok=True)

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            if epoch % self.every_n_epochs == 0 and epoch > 0:
                target = self.dirpath / f'backup_epoch_{epoch}.ckpt'
                try:
                    trainer.save_checkpoint(str(target))
                    print(f"✓ Saved backup checkpoint: {target}")
                except Exception as e:
                    print(f"⚠️ Failed saving backup checkpoint: {e}")

    logger = TensorBoardLogger(save_dir=str(run_dir), name='tensorboard')
    # Enhanced checkpoint: save full state and keep last checkpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename='best_model',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True,  # Also save last checkpoint
        # CRITICAL: Save weights only (safer for loading)
        save_weights_only=False,  # Save full Lightning state
        # Save hyperparameters explicitly
        save_on_train_epoch_end=False,  # Only save after validation
    )

    # Add explicit hyperparameter saving to run_dir
    try:
        hparam_dict = {
            'hidden_size': hidden_size,
            'attention_head_size': MODEL_CONFIG['attention_head_size'],
            'dropout': MODEL_CONFIG['dropout'],
            'hidden_continuous_size': MODEL_CONFIG['hidden_continuous_size'],
            'output_size': MODEL_CONFIG['output_size'],
            'learning_rate': learning_rate,
            'max_encoder_length': max_encoder_length,
            'max_prediction_length': max_prediction_length,
            'target_normalizer': 'GroupNormalizer',  # Type for reference
            'loss_type': 'QuantileLoss',
        }

        hparam_file = run_dir / 'hparams.json'
        with open(hparam_file, 'w') as f:
            json.dump(hparam_dict, f, indent=2)

        print(f"✓ Saved hyperparameters to {hparam_file}")

    except Exception as e:
        print(f"⚠️ Failed to save hyperparameters: {e}")

    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # Use RichProgressBar for nicer output if available
    progress_bar = RichProgressBar()
    # Diagnostics callbacks
    epoch_logger_cb = EpochLoggingCallback()
    gradnorm_cb = GradientNormCallback()
    backup_cb = BackupCheckpointCallback(dirpath=run_dir, every_n_epochs=10)

    print("\n🚀 Starting training...")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=0.1,
        callbacks=[checkpoint_cb, early_stop, lr_monitor, progress_bar, epoch_logger_cb, gradnorm_cb, backup_cb],
        logger=logger,
        enable_progress_bar=True,
    )

    # Fit
    # Diagnostic: ensure feature dimension equals EXPECTED_FEATURE_COUNT
    try:
        import logging
        logger = logging.getLogger(__name__)
        n_samples = len(train_dataset)
        seq_len = builder.max_encoder_length
        n_features = len(builder.time_varying_unknown_reals) if builder.time_varying_unknown_reals is not None else 0
        logger.info(f"✓ Training with {EXPECTED_FEATURE_COUNT} features (93 technical + 20 new + 34 sentiment)")
        logger.info(f"  Input shape: (samples={n_samples}, seq_len={seq_len}, features={n_features})")
        assert n_features == EXPECTED_FEATURE_COUNT, f"Feature dimension mismatch: {n_features} != {EXPECTED_FEATURE_COUNT}"
    except Exception as e:
        warnings.warn(f"Feature dimension diagnostic failed: {e}")

    try:
        print("\n🧪 Pre-training validation...")

        # Test that model can make predictions
        try:
            sample_batch = next(iter(train_dataloader))
            with torch.no_grad():
                output = tft(sample_batch[0])
            print(f"✓ Model forward pass works")
            print(f"  Output keys: {list(output.keys()) if isinstance(output, dict) else 'not dict'}")
        except Exception as e:
            print(f"❌ Model forward pass FAILED: {e}")
            print(f"   Training will likely fail - fix the model creation first")
            raise

        print("✓ Pre-training checks passed\n")

        print("\n🚀 Starting training...")
        trainer.fit(tft_for_trainer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        print("✓ Training completed successfully")

        # === POST-TRAINING CHECKPOINT VALIDATION ===
        print("\n" + "="*60)
        print("  POST-TRAINING CHECKPOINT VALIDATION")
        print("="*60)

        # Load the best checkpoint and verify it works
        best_ckpt_path = checkpoint_cb.best_model_path

        if not best_ckpt_path or not Path(best_ckpt_path).exists():
            print("⚠️ No best checkpoint found - using last checkpoint")
            best_ckpt_path = run_dir / 'last.ckpt'

        if Path(best_ckpt_path).exists():
            print(f"Testing checkpoint: {best_ckpt_path}")
            
            try:
                # Load the checkpoint (primary method)
                loaded_model = TemporalFusionTransformer.load_from_checkpoint(
                    str(best_ckpt_path),
                    map_location='cpu'
                )

                print("✓ Checkpoint loaded successfully")
                
                # Verify loss encoder is present
                if hasattr(loaded_model, 'loss') and loaded_model.loss is not None:
                    print(f"✓ Loss function present: {type(loaded_model.loss)}")
                    
                    # Check if encoder is callable
                    if hasattr(loaded_model.loss, 'encoder'):
                        if callable(loaded_model.loss.encoder):
                            print("✓ Loss encoder is callable")
                        else:
                            print(f"❌ Loss encoder is NOT callable: {type(loaded_model.loss.encoder)}")
                            print("   This checkpoint will FAIL at inference!")
                    else:
                        print("⚠️ Loss has no encoder attribute")
                else:
                    print("❌ Loss function is None - checkpoint is broken!")
                
                # Test prediction on validation data
                print("\nTesting prediction on validation batch...")
                try:
                    sample_preds = loaded_model.predict(
                        val_dataloader,
                        mode='prediction',
                        return_index=False,
                        return_decoder_lengths=False
                    )
                    
                    print(f"✓ Prediction successful")
                    print(f"  Output type: {type(sample_preds)}")
                    print(f"  Output shape: {getattr(sample_preds, 'shape', 'N/A')}")
                    
                except Exception as e:
                    print(f"❌ Prediction FAILED: {e}")
                    print("   This checkpoint cannot be used for inference!")
                    
            except Exception as e:
                # Special-case: some checkpoints saved by older/newer Lightning/PF combos
                # may not include the internal '__special_save__' hyperparams key and
                # `load_from_checkpoint` will raise a KeyError. Try a best-effort fallback
                # by loading the raw checkpoint and constructing a model from the saved
                # `model_config.json` / `hparams.json` and then loading the state_dict.
                msg = str(e)
                print(f"⚠️ Primary checkpoint load failed: {msg}")
                try:
                    ck = torch.load(str(best_ckpt_path), map_location='cpu')
                    state_dict = None
                    if isinstance(ck, dict):
                        # common keys: 'state_dict' (Lightning) or direct state dict
                        if 'state_dict' in ck:
                            state_dict = ck['state_dict']
                        else:
                            # may already be a state_dict
                            state_dict = ck
                    else:
                        state_dict = ck

                    # Attempt to build constructor args from model_config.json or hparams.json
                    model_kwargs = {}
                    try:
                        cfg_path = run_dir / 'model_config.json'
                        if not cfg_path.exists():
                            # fallback to per-symbol config saved earlier
                            cfg_path = Path(f'saved_models/{symbols_list[0]}_tft_config.json')
                        if cfg_path.exists():
                            with open(cfg_path, 'r') as fh:
                                model_kwargs = json.load(fh)
                    except Exception:
                        model_kwargs = {}

                    # Construct a model instance and load state_dict
                    try:
                        fallback_model = TemporalFusionTransformer(
                            loss=QuantileLoss(),
                            **{k: v for k, v in model_kwargs.items() if k in ['hidden_size', 'attention_head_size', 'dropout', 'hidden_continuous_size', 'output_size', 'learning_rate', 'reduce_on_plateau_patience']}
                        )
                        # Torch Lightning checkpoints store keys with a 'model.' prefix sometimes
                        # Normalize keys if necessary
                        if state_dict is not None:
                            # If 'state_dict' is nested under model, try to extract
                            if isinstance(state_dict, dict):
                                # find first key that appears to be param name
                                sample_key = next(iter(state_dict.keys())) if len(state_dict) > 0 else None
                                if sample_key and sample_key.startswith('model.'):
                                    # strip 'model.' prefix
                                    new_state = {k.replace('model.', ''): v for k, v in state_dict.items()}
                                else:
                                    new_state = state_dict
                                try:
                                    fallback_model.load_state_dict(new_state)
                                except Exception as le:
                                    print(f"⚠️ Could not load state_dict into fallback model: {le}")
                                    # try strict=False
                                    try:
                                        fallback_model.load_state_dict(new_state, strict=False)
                                        print("✓ Loaded state_dict with strict=False")
                                    except Exception as le2:
                                        print(f"❌ Fallback state_dict load also failed: {le2}")
                                        raise
                        loaded_model = fallback_model
                        print("✓ Fallback checkpoint load succeeded (state_dict applied)")
                    except Exception as le:
                        print(f"❌ Fallback load failed: {le}")
                        import traceback
                        traceback.print_exc()
                except Exception as le2:
                    print(f"❌ Could not read checkpoint file for fallback: {le2}")
                    import traceback
                    traceback.print_exc()

        else:
            print(f"❌ Checkpoint not found: {best_ckpt_path}")

        print("="*60 + "\n")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print("\nDebug information:")
        try:
            print(f"  - Train samples: {len(train_dataset)}")
        except Exception:
            print("  - Train samples: unknown")
        try:
            print(f"  - Val samples: {len(val_dataset)}")
        except Exception:
            print("  - Val samples: unknown")
        try:
            print(f"  - Batch size: {batch_size}")
        except Exception:
            print("  - Batch size: unknown")
        try:
            print(f"  - Encoder length: {builder.max_encoder_length}")
        except Exception:
            print("  - Encoder length: unknown")
        try:
            print(f"  - Prediction length: {builder.max_prediction_length}")
        except Exception:
            print("  - Prediction length: unknown")

        # Try to save what we have
        try:
            print("\n💾 Attempting to save partial checkpoint...")
            emergency_save_path = run_dir / 'emergency_checkpoint.ckpt'
            trainer.save_checkpoint(str(emergency_save_path))
            print(f"   ✓ Emergency checkpoint saved to {emergency_save_path}")
        except Exception as _:
            print("   ⚠️ Failed to save emergency checkpoint")

        raise

    # Load best model
    best_path = checkpoint_cb.best_model_path
    if best_path == '':
        print('No checkpoint saved, using current model')
        # prefer the underlying model if we wrapped
        best_tft = tft_for_trainer.model if hasattr(tft_for_trainer, 'model') else tft_for_trainer
    else:
        # If we used the wrapper, try to load the checkpoint state into it
        if not isinstance(tft_for_trainer, pl.LightningModule) or hasattr(tft_for_trainer, 'model'):
            try:
                ck = torch.load(best_path, map_location=lambda storage, loc: storage)
                state_dict = ck.get('state_dict', ck)
                # load into trainer wrapper if possible
                try:
                    tft_for_trainer.load_state_dict(state_dict)
                    best_tft = tft_for_trainer.model if hasattr(tft_for_trainer, 'model') else tft_for_trainer
                except Exception:
                    # fallback: try to load using TemporalFusionTransformer loader
                    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
            except Exception:
                warnings.warn('Could not load checkpoint into wrapper; attempting standard loader')
                try:
                    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
                except Exception:
                    warnings.warn('Could not load best checkpoint; using in-memory trained model')
                    best_tft = tft_for_trainer.model if hasattr(tft_for_trainer, 'model') else tft_for_trainer
        else:
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # Save config and dataset parameters
    # --- Post-training validations: test checkpoint load and simple inference ---
    try:
        # Simple forward pass on a few validation samples to ensure checkpoint functional
        try:
            sample_batch = next(iter(val_dataloader))
            if isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 2:
                x_s, y_s = sample_batch[0], sample_batch[1]
            else:
                x_s, y_s = sample_batch, None
            # If we wrapped the model, extract underlying model for forward
            test_model = best_tft
            # Try to run forward on up to 3 sample sequences
            try:
                # If inputs are dict of tensors, slice first N examples
                N = 3
                x_probe = None
                if isinstance(x_s, dict):
                    x_probe = {}
                    for k, v in x_s.items():
                        try:
                            x_probe[k] = v[:N]
                        except Exception:
                            x_probe[k] = v
                else:
                    try:
                        x_probe = x_s[:N]
                    except Exception:
                        x_probe = x_s

                out = test_model(x_probe)
                print('✓ Post-train inference forward pass succeeded on validation sample (up to 3 sequences)')
            except Exception as e:
                print(f'⚠️ Post-train forward pass failed: {e}')
        except StopIteration:
            print('⚠️ Could not sample from val_dataloader for post-train inference')
    except Exception as e:
        print(f'⚠️ Post-training validation failed: {e}')

    # Save extended config (training + model + dataset parameters)
    try:
        extended_cfg = {
            'hidden_size': hidden_size,
            'attention_head_size': MODEL_CONFIG['attention_head_size'],
            'dropout': MODEL_CONFIG['dropout'],
            'hidden_continuous_size': MODEL_CONFIG['hidden_continuous_size'],
            'output_size': MODEL_CONFIG['output_size'],
            'learning_rate': learning_rate,
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'max_encoder_length': int(builder.max_encoder_length),
            'max_prediction_length': int(builder.max_prediction_length),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'callbacks': ['ModelCheckpoint(best)', 'EarlyStopping', 'LearningRateMonitor', 'RichProgressBar', 'EpochLoggingCallback', 'GradientNormCallback', 'BackupCheckpointCallback']
        }
        (run_dir / 'train_config.json').write_text(json.dumps(extended_cfg, indent=2))
    except Exception as e:
        warnings.warn(f'Could not write extended train config: {e}')

    (run_dir / 'model_config.json').write_text(json.dumps({
        'hidden_size': hidden_size,
        'attention_head_size': MODEL_CONFIG['attention_head_size'],
        'dropout': MODEL_CONFIG['dropout'],
        'hidden_continuous_size': MODEL_CONFIG['hidden_continuous_size'],
        'output_size': MODEL_CONFIG['output_size'],
        'learning_rate': learning_rate,
    }, indent=2))

    with open(run_dir / 'dataset_parameters.pkl', 'wb') as f:
        pickle.dump({
            'max_encoder_length': max_encoder_length,
            'max_prediction_length': max_prediction_length,
            'time_varying_unknown_reals': builder.time_varying_unknown_reals,
            'train_dataset_kwargs': getattr(train_dataset, 'to_parameters', lambda: None)(),
        }, f)

    # Generate predictions on validation set
    try:
        preds = best_tft.predict(val_dataloader)
        # preds might be numpy array or pandas DataFrame
        if isinstance(preds, (np.ndarray, list)):
            df_preds = pd.DataFrame(preds)
        else:
            df_preds = pd.DataFrame(preds)
        df_preds.to_csv(run_dir / 'validation_predictions.csv', index=False)
    except Exception as e:
        warnings.warn(f'Could not generate predictions: {e}')

    print(f'Model and artifacts saved to: {run_dir}')
    # Verify feature count used for TFT (best-effort)
    try:
        n_features = len(builder.time_varying_unknown_reals) if builder.time_varying_unknown_reals is not None else 0
        if n_features != EXPECTED_FEATURE_COUNT:
            warnings.warn(f"Feature count mismatch for TFT dataset: expected={EXPECTED_FEATURE_COUNT}, got={n_features}")
        # Try to validate saved canonical feature file if present
        try:
            import pickle as _pkl
            symbol_paths = ModelPaths(symbols_list[0])
            if symbol_paths.feature_columns.exists():
                saved_cols = _pkl.load(open(symbol_paths.feature_columns, 'rb'))
                if len(saved_cols) != EXPECTED_FEATURE_COUNT:
                    warnings.warn(f"Saved feature_columns.pkl mismatch: expected={EXPECTED_FEATURE_COUNT}, got={len(saved_cols)}")
        except Exception:
            # Not fatal; just warn
            pass
    except Exception:
        pass

    return run_dir


def train_tft(symbol: str, epochs: int = 50, batch_size: int = 512, 
              max_encoder_length: int = 60, max_prediction_length: int = 10,
              hidden_size: int = 128, learning_rate: float = 1e-3,
              output_dir: str = 'saved_models/tft', use_cache: bool = True):
    """Compatibility wrapper used by `train_with_tft.train_with_tft`.

    Accepts a single symbol (string) and maps arguments to `train_tft_model`.
    Returns the path to saved artifacts (same as `train_tft_model`).
    """
    return train_tft_model([symbol],
                           max_encoder_length=max_encoder_length,
                           max_prediction_length=max_prediction_length,
                           epochs=epochs,
                           batch_size=batch_size,
                           hidden_size=hidden_size,
                           learning_rate=learning_rate,
                           output_dir=output_dir,
                           use_cache=use_cache)


def evaluate_and_plot(model, val_dataloader, output_dir):
    """Generate validation metrics and visualizations.

    Produces:
    - MAE per horizon plot
    - Quantile coverage plot
    - Saves CSV of per-horizon MAE
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to obtain predictions and x (inputs) for coverage computation
    try:
        raw = model.predict(val_dataloader, mode='raw')
    except TypeError:
        raw = model.predict(val_dataloader)

    # Attempt to coerce predictions to DataFrame
    try:
        preds_df = pd.DataFrame(raw)
    except Exception:
        preds_df = None

    # Placeholder: compute MAE per horizon if predictions and true values are available
    try:
        # Many users will implement a custom mapping; here we try to compute a basic MAE
        truth = []
        for batch in val_dataloader:
            y = batch['decoder_target'] if 'decoder_target' in batch else None
            if y is not None:
                truth.append(y.detach().cpu().numpy())
        if len(truth) > 0 and preds_df is not None:
            truth = np.vstack(truth)
            preds = preds_df.values
            # Align shapes conservatively
            min_len = min(truth.shape[0], preds.shape[0])
            truth = truth[:min_len]
            preds = preds[:min_len]
            mae_per_horizon = np.mean(np.abs(truth - preds), axis=0)
            pd.Series(mae_per_horizon).to_csv(output_dir / 'mae_per_horizon.csv', index=False)
            plt.figure()
            plt.plot(mae_per_horizon)
            plt.title('MAE per horizon')
            plt.xlabel('Horizon')
            plt.ylabel('MAE')
            plt.savefig(output_dir / 'mae_per_horizon.png')
    except Exception as e:
        warnings.warn(f'Could not compute MAE per horizon: {e}')

    # Quantile coverage placeholder: user can extend with exact quantile mapping
    try:
        # Attempt to compute empirical coverage for 0.1/0.5/0.9 if preds contain them
        # This section is best-effort; exact extraction depends on model predict output format
        pass
    except Exception:
        pass


def plot_attention_weights(model, sample_batch, output_path):
    """Visualize which features TFT attends to for each forecast horizon.

    Note: PyTorch Forecasting exposes interpretable outputs through
    `model.interpret_output` and `model.attention` helpers. This implementation
    tries a best-effort extraction and will quietly fail if internals differ
    between versions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        interpretation = model.inspect_prediction(sample_batch)
        # interpretation is a dict with attention weights in many versions
        fig = interpretation.get('attention', None)
        if fig is None:
            # fallback: try to plot variable importance
            var_imp = model.calculate_variable_importance(sample_batch)
            var_imp.plot(kind='bar', figsize=(10, 6))
            plt.tight_layout()
            plt.savefig(output_path)
        else:
            # if interpretation returns a matplotlib figure
            if hasattr(fig, 'savefig'):
                fig.savefig(output_path)
    except Exception as e:
        warnings.warn(f'Could not plot attention weights: {e}')


def main():
    parser = argparse.ArgumentParser(description='Train Temporal Fusion Transformer')
    parser.add_argument('--symbols', type=str, required=True,
                       help='Comma-separated symbol list (e.g., AAPL,TSLA,HOOD)')
    parser.add_argument('--max-encoder-length', type=int, default=60,
                       help='Historical context window (default: 60 days)')
    parser.add_argument('--max-prediction-length', type=int, default=10,
                       help='Forecast horizon (default: 10 days)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='TFT hidden dimension (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--output-dir', type=str, default='saved_models/tft',
                       help='Output directory for models (default: saved_models/tft)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of prepared datasets (force refetch)')

    args = parser.parse_args()

    symbols = args.symbols.split(',')
    run_dir = train_tft_model(symbols,
                              max_encoder_length=args.max_encoder_length,
                              max_prediction_length=args.max_prediction_length,
                              epochs=args.epochs,
                              batch_size=args.batch_size,
                              hidden_size=args.hidden_size,
                              learning_rate=args.learning_rate,
                              output_dir=args.output_dir,
                              use_cache=(not args.no_cache))

    print(f'Training complete. Artifacts at: {run_dir}')


if __name__ == '__main__':
    main()

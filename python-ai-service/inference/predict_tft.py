import torch
import pandas as pd
import numpy as np
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import warnings
import logging
import io
import contextlib

# Import ModelPaths for organized path management
try:
    from utils.model_paths import ModelPaths, get_legacy_tft_paths
except ImportError:
    ModelPaths = None
    get_legacy_tft_paths = None

# Suppress noisy logs/warnings from PyTorch/PyTorch-Forecasting during checkpoint loads
logging.getLogger('pytorch_forecasting').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
from pytorch_forecasting.metrics import QuantileLoss

# Module logger: default to WARNING to keep TFT loading quiet in normal runs
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def ensure_loss_encoder(model):
    """Attach a safe encoder/rescale wrapper to model.loss when missing.

    Some checkpoints save QuantileLoss without the runtime encoder callable
    that `rescale_parameters` expects. That results in a TypeError during
    prediction. This helper attempts to (1) set a best-effort `loss.encoder`
    callable and (2) wrap `loss.rescale_parameters` to avoid calling a None
    encoder. Both operations are best-effort and will not raise.
    """
    try:
        loss = getattr(model, 'loss', None)
        if loss is None:
            return

        # best-effort encoder that returns prediction when target_scale absent
        def _safe_encoder(d):
            try:
                if not isinstance(d, dict):
                    return d
                pred = d.get('prediction', None)
                target_scale = d.get('target_scale', None)
                if pred is None:
                    return d
                if target_scale is None:
                    return pred
                # try simple elementwise scaling if possible
                try:
                    import torch
                    if hasattr(pred, 'shape') and hasattr(target_scale, 'shape'):
                        try:
                            return pred * target_scale
                        except Exception:
                            return pred
                except Exception:
                    pass
                if isinstance(target_scale, dict):
                    sc = target_scale.get('scale', None)
                    if sc is not None:
                        try:
                            return pred * sc
                        except Exception:
                            return pred
                return pred
            except Exception:
                return d

        try:
            if not hasattr(loss, 'encoder') or getattr(loss, 'encoder') is None:
                loss.encoder = _safe_encoder
        except Exception:
            pass

        # Wrap rescale_parameters to avoid calling None encoder; idempotent
        try:
            import types
            if hasattr(loss, 'rescale_parameters') and not getattr(loss, '_rescale_wrapped', False):
                orig = loss.rescale_parameters

                def _safe_rescale(self, parameters, target_scale=None, **kwargs):
                    try:
                        enc = getattr(self, 'encoder', None)
                        if enc is None:
                            # fallback: attempt to scale parameters by target_scale if numeric
                            if target_scale is None:
                                return parameters
                            try:
                                import torch
                                if hasattr(parameters, 'shape') and hasattr(target_scale, 'shape'):
                                    try:
                                        return parameters * target_scale
                                    except Exception:
                                        return parameters
                            except Exception:
                                pass
                            if isinstance(target_scale, dict):
                                sc = target_scale.get('scale', None)
                                if sc is not None:
                                    try:
                                        return parameters * sc
                                    except Exception:
                                        return parameters
                            return parameters
                        try:
                            return orig(parameters, target_scale, **kwargs)
                        except TypeError:
                            return orig(parameters, target_scale)
                    except Exception:
                        return parameters

                loss.rescale_parameters = types.MethodType(_safe_rescale, loss)
                setattr(loss, '_rescale_wrapped', True)
        except Exception:
            pass
    except Exception:
        return


def strip_checkpoint_prefixes(state_dict: dict) -> dict:
    """Remove leading 'model.' and 'module.' prefixes from checkpoint keys.

    This removes a single leading occurrence of these prefixes when present.
    It preserves other parts of the key and returns a new dict mapping.
    """
    prefixes = ("model.", "module.")
    fixed = {}
    for k, v in state_dict.items():
        new_k = k
        # remove any of the common prefixes if they appear at the start
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
                # only remove one prefix occurrence (do not loop-remove repeatedly)
                break
        fixed[new_k] = v
    return fixed

def load_state_dict_quietly(model, state_dict, strict=True):
    """Load a state_dict into `model` while capturing large stderr output.

    Returns: (missing_keys, unexpected_keys, error_message_or_None)
    """
    import io, contextlib
    stderr_capture = io.StringIO()
    try:
        # Redirect stderr to capture the large PyTorch error dump
        with contextlib.redirect_stderr(stderr_capture):
            res = model.load_state_dict(state_dict, strict=strict)

        # Normalize return shape
        missing = []
        unexpected = []
        if hasattr(res, 'missing_keys') or hasattr(res, 'unexpected_keys'):
            missing = getattr(res, 'missing_keys', []) or []
            unexpected = getattr(res, 'unexpected_keys', []) or []
        elif isinstance(res, tuple) and len(res) >= 2:
            missing, unexpected = res[0], res[1]

        return missing, unexpected, None

    except RuntimeError as e:
        # Only return the first line to avoid massive dumps
        msg = str(e).split('\n')[0]
        return [], [], msg


def load_tft_checkpoint_with_key_fix(checkpoint_path, model):
    """Load checkpoint file and apply key-fix loader to the model."""
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        logger.error("Failed to read checkpoint: %s", e)
        return None

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        logger.error("Unsupported checkpoint format for %s", checkpoint_path)
        return None

    # Strip only a single leading 'model.' prefix for cleanliness
    fixed_state = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_k = k[len('model.'):]
        else:
            new_k = k
        fixed_state[new_k] = v

    try:
        # Capture stdout/stderr to avoid printing massive key lists
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            load_res = model.load_state_dict(fixed_state, strict=False)
        # Extract missing/unexpected keys in a robust way
        if hasattr(load_res, 'missing_keys') or hasattr(load_res, 'unexpected_keys'):
            missing = getattr(load_res, 'missing_keys', []) or []
            unexpected = getattr(load_res, 'unexpected_keys', []) or []
        elif isinstance(load_res, tuple) and len(load_res) >= 2:
            missing, unexpected = load_res[0], load_res[1]
        else:
            missing, unexpected = [], []

        # Summarize instead of dumping long key lists
        if missing:
            logger.debug("%d keys initialized randomly (not in checkpoint)", len(missing))
        if unexpected:
            logger.debug("%d checkpoint keys not used by model", len(unexpected))

        # Heuristic: if more than half of model keys are missing, treat as failure
        if len(missing) > max(1, int(len(fixed_state) * 0.5)):
            logger.warning(">50%% of model keys missing from checkpoint - architecture mismatch suspected")
            return None

        return model
    except Exception as e:
        msg = str(e)
        logger.error("Failed to load state dict: %s", msg[:200])
        return None


def load_tft_dataset(symbol: str, max_encoder_length: int = 60, max_prediction_length: int = 5, include_sentiment: bool = True):
    """Build a minimal TimeSeriesDataSet for `symbol` to instantiate a TFT via `from_dataset`.

    This uses `prepare_tft_data_for_inference` to generate a dataframe and then
    constructs a TimeSeriesDataSet with reasonable defaults required by TFT.
    """
    # prepare dataframe
    df = prepare_tft_data_for_inference(symbol=symbol, include_sentiment=include_sentiment)

    # Determine feature columns (exclude known columns)
    exclude = ['time_idx', 'symbol', 'returns', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [c for c in df.columns if c not in exclude]

    # Create TimeSeriesDataSet
    dataset = TimeSeriesDataSet(
        df,
        time_idx='time_idx',
        target='returns',
        group_ids=['symbol'],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=feature_cols + ['returns'],
        target_normalizer=GroupNormalizer(groups=['symbol'])
    )

    return dataset


# Simple in-memory cache to avoid repeated expensive data fetches
_tft_dataset_cache = {}

def get_cached_tft_dataset(symbol: str, max_encoder_length: int = 60, max_prediction_length: int = 5, include_sentiment: bool = True):
    key = (symbol.upper() if symbol else symbol, int(max_encoder_length), int(max_prediction_length), bool(include_sentiment))
    if key in _tft_dataset_cache:
        return _tft_dataset_cache[key]
    ds = load_tft_dataset(symbol, max_encoder_length=max_encoder_length, max_prediction_length=max_prediction_length, include_sentiment=include_sentiment)
    _tft_dataset_cache[key] = ds
    return ds


def load_tft_model_from_symbol(symbol: str, checkpoint_path: str | None = None):
    """Load TFT model by symbol, extracting architecture from checkpoint when possible.

    Returns (model, ds_params)
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        # Check new path structure first, then legacy
        checkpoint_candidates = []
        if ModelPaths is not None:
            paths = ModelPaths(symbol)
            checkpoint_candidates.append(paths.tft.checkpoint)
        # Legacy paths
        checkpoint_candidates.extend([
            Path('saved_models') / symbol / 'tft' / 'best_model.ckpt',
            Path('saved_models') / 'tft' / symbol / 'best_model.ckpt',
            Path('saved_models') / f'{symbol}_tft.ckpt',
        ])
        
        checkpoint_path = None
        for candidate in checkpoint_candidates:
            if candidate.exists():
                checkpoint_path = candidate
                break
        
        if checkpoint_path is None:
            checkpoint_path = Path('saved_models') / f'{symbol}_tft.ckpt'  # Fallback for error message

    # Validate checkpoint and extract hidden size
    info = validate_tft_checkpoint(str(checkpoint_path))
    if not info.get('valid'):
        raise ValueError(f"Cannot load TFT checkpoint: {info.get('error')}")

    hidden_size = info.get('hidden_size') or 128
    attention_head_size = max(1, hidden_size // 4)

    # Attempt to build a dataset to infer input feature structure
    try:
        cfg = load_tft_hyperparameters(str(checkpoint_path))
        max_encoder = cfg.get('max_encoder_length', 60)
        max_prediction = cfg.get('max_prediction_length', 5)
    except Exception:
        max_encoder = 60
        max_prediction = 5

    dataset = load_tft_dataset(symbol, max_encoder_length=max_encoder, max_prediction_length=max_prediction)

    # Instantiate model with detected hidden size
    tft_model = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=0.1,
        hidden_continuous_size=max(8, hidden_size // 2),
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # Load weights with key-fix
    tft_model = load_tft_checkpoint_with_key_fix(str(checkpoint_path), tft_model)

    ds_params = {
        'max_encoder_length': max_encoder,
        'max_prediction_length': max_prediction
    }
    return tft_model, ds_params



def load_tft_model_for_inference(symbol, checkpoint_path=None):
    """Load TFT model with proper key fixing."""
    import torch
    from pytorch_forecasting import TemporalFusionTransformer
    
    if checkpoint_path is None:
        # Check new path structure first, then legacy
        checkpoint_candidates = []
        if ModelPaths is not None:
            paths = ModelPaths(symbol)
            checkpoint_candidates.append(paths.tft.checkpoint)
        # Legacy paths
        checkpoint_candidates.extend([
            Path(f'saved_models/{symbol}/tft/best_model.ckpt'),
            Path(f'saved_models/tft/{symbol}/best_model.ckpt'),
            Path(f'saved_models/tft/{symbol}/best_model-v9.ckpt'),
            Path(f'saved_models/{symbol}_tft.ckpt'),
        ])
        
        checkpoint_path = None
        for candidate in checkpoint_candidates:
            if candidate.exists():
                checkpoint_path = str(candidate)
                break
        
        if checkpoint_path is None:
            checkpoint_path = f'saved_models/tft/{symbol}/best_model-v9.ckpt'  # Fallback for error
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Validate checkpoint format
    checkpoint_info = validate_tft_checkpoint(checkpoint_path)
    logger.info("TFT Checkpoint Info:")
    logger.info("  Format: %s", checkpoint_info.get('format'))
    logger.info("  Has 'model.' prefix: %s", checkpoint_info.get('has_model_prefix'))
    logger.info("  Detected hidden_size: %s", checkpoint_info.get('hidden_size'))
    logger.info("  Total keys: %s", checkpoint_info.get('total_keys'))
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        original_state_dict = checkpoint['state_dict']
    else:
        original_state_dict = checkpoint
    # keep original_state_dict as loaded; we will try loading with original keys first
    # detect common prefixes for logging
    has_model_prefix = any(k.startswith('model.') for k in original_state_dict.keys())
    has_module_prefix = any(k.startswith('module.') for k in original_state_dict.keys())
    logger.debug("Checkpoint key prefixes -> model.: %s, module.: %s", has_model_prefix, has_module_prefix)

    # prepare a 'fixed' variant with common prefixes stripped for later heuristics
    fixed_state = strip_checkpoint_prefixes(original_state_dict if isinstance(original_state_dict, dict) else {})

    # Load dataset to get structure (use cached helper to avoid refetch)
    try:
        dataset = get_cached_tft_dataset(symbol, max_encoder_length=checkpoint_info.get('max_encoder_length', 60),
                                         max_prediction_length=checkpoint_info.get('max_prediction_length', 5))
    except Exception as e:
        logger.error("Failed to load TFT dataset: %s", e)
        return None, {}

    # Create model with correct hidden_size
    hidden_size = checkpoint_info.get('hidden_size', 128)

    try:
        # Create model structure
        tft_model = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=hidden_size,
            attention_head_size=hidden_size // 4,
            dropout=0.1,
            hidden_continuous_size=hidden_size // 2,
            output_size=7,
            learning_rate=0.001,
        )

        # --- New robust loading strategy: try original keys, then stripped prefixes ---
        try:
            # Attempt A: load using original checkpoint keys
            missing, unexpected, error = load_state_dict_quietly(tft_model, original_state_dict, strict=False)
            if error is None and (len(missing) <= max(1, int(len(original_state_dict) * 0.5))):
                logger.info("TFT model loaded using original checkpoint keys (missing=%d, unexpected=%d)", len(missing), len(unexpected))
                ds_params = {'max_encoder_length': checkpoint_info.get('max_encoder_length', 60),
                             'max_prediction_length': checkpoint_info.get('max_prediction_length', 5)}
                return tft_model, ds_params
            else:
                logger.debug("Original-key load: error=%s, missing=%d, unexpected=%d", error, len(missing), len(unexpected))
        except Exception as _:
            logger.debug("Original-key load attempt raised an exception; continuing to prefix-stripped attempt")

        try:
            # Attempt B: strip common prefixes and retry
            stripped_state = strip_checkpoint_prefixes(original_state_dict)
            missing, unexpected, error = load_state_dict_quietly(tft_model, stripped_state, strict=False)
            if error is None and (len(missing) <= max(1, int(len(stripped_state) * 0.5))):
                logger.info("TFT model loaded after stripping prefixes (missing=%d, unexpected=%d)", len(missing), len(unexpected))
                ds_params = {'max_encoder_length': checkpoint_info.get('max_encoder_length', 60),
                             'max_prediction_length': checkpoint_info.get('max_prediction_length', 5)}
                return tft_model, ds_params
            else:
                logger.debug("Stripped-key load: error=%s, missing=%d, unexpected=%d", error, len(missing), len(unexpected))
        except Exception as _:
            logger.debug("Stripped-key load attempt raised an exception; continuing to other strategies")

        # Strategy 1: try adaptive remapping (best-effort)
        try:
            remapped, version_info = adapt_tft_checkpoint_keys(checkpoint, tft_model)
            missing_keys, unexpected_keys, error = load_state_dict_quietly(tft_model, remapped, strict=False)
            if error is None:
                ds_params = {'max_encoder_length': checkpoint_info.get('max_encoder_length', 60),
                             'max_prediction_length': checkpoint_info.get('max_prediction_length', 5)}
                logger.info("TFT model loaded successfully with adaptive remapping: %s", version_info)
                return tft_model, ds_params
        except Exception:
            # continue to next strategy
            pass

        # Strategy 2: try the existing heuristic loader which applies key fixes
        try:
            loaded = load_tft_checkpoint_with_key_fix(checkpoint_path, tft_model)
            if loaded is not None:
                ds_params = {'max_encoder_length': checkpoint_info.get('max_encoder_length', 60),
                             'max_prediction_length': checkpoint_info.get('max_prediction_length', 5)}
                logger.info("TFT model loaded successfully using heuristic key-fix loader")
                return loaded, ds_params
        except Exception:
            pass

        # Strategy 3: final fallback - try filtered assignment of matching-shaped params
        model_state = tft_model.state_dict()
        filtered = {}
        for k, v in fixed_state.items():
            if k in model_state:
                try:
                    if tuple(model_state[k].shape) == tuple(v.shape):
                        filtered[k] = v
                except Exception:
                    continue

        if filtered:
            missing_keys, unexpected_keys, error = load_state_dict_quietly(tft_model, filtered, strict=False)
            if error is None:
                loaded_count = len(filtered)
                logger.info("Partially loaded %d matching checkpoint keys into TFT model (filtered)", loaded_count)
                ds_params = {'max_encoder_length': checkpoint_info.get('max_encoder_length', 60),
                             'max_prediction_length': checkpoint_info.get('max_prediction_length', 5)}
                return tft_model, ds_params

        # If filtered approach didn't work, fall back to naive attempt (may still fail)
        missing_keys, unexpected_keys, error = load_state_dict_quietly(tft_model, fixed_state, strict=False)
        if error:
            logger.error("TFT state dict loading failed: %s", error)
            return None, {}

        # Only show summary, not full list
        if missing_keys:
            logger.debug("%d model keys not in checkpoint (using random init)", len(missing_keys))
        if unexpected_keys:
            logger.debug("%d checkpoint keys not used", len(unexpected_keys))

        # Check if too many keys are missing (indicates major mismatch)
        total_model_keys = len(tft_model.state_dict())
        if len(missing_keys) > total_model_keys * 0.3:
            logger.warning("%d/%d keys missing (>30%%) - model may not work correctly; consider retraining TFT", len(missing_keys), total_model_keys)

        logger.info("TFT model loaded successfully (naive strip)")
        ds_params = {'max_encoder_length': checkpoint_info.get('max_encoder_length', 60),
                     'max_prediction_length': checkpoint_info.get('max_prediction_length', 5)}
        return tft_model, ds_params

    except Exception as e:
        print(f"❌ Failed to create/load TFT model: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None, {}


def load_tft_model(checkpoint_path: str, symbol: str = None, map_location='cpu'):
    """Load TFT model with robust fallback strategies.
    
    Args:
        checkpoint_path: Path to .ckpt file
        symbol: EXPLICIT symbol (e.g., 'AAPL') - DO NOT INFER FROM PATH
        map_location: Device for model loading
    
    Returns:
        (model, dataset_params) or (None, {}) if load fails
    """
    from pathlib import Path
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
    import json
    import re
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return None, {}
    
    logger.info("Loading TFT checkpoint: %s", checkpoint_path)
    
    # Load checkpoint once and prepare common state dict variants
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        logger.error("Failed to read checkpoint: %s", e)
        return None, {}

    # canonical state dict objects
    original_state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # stripped- prefix variant for many checkpoints
    stripped_state_dict = strip_checkpoint_prefixes(original_state_dict if isinstance(original_state_dict, dict) else {})

    # Lightweight runtime validator used by all strategies: returns True if a small
    # dataloader prediction completes and returns non-None (no exceptions).
    def _validate_model_predict(candidate_model, params):
        try:
            # Ensure the model.loss has a safe encoder to avoid NoneType callables
            try:
                ensure_loss_encoder(candidate_model)
            except Exception:
                pass
            ds_try = None
            dl_try = None
            try:
                ds_try = get_cached_tft_dataset(symbol, max_encoder_length=params.get('max_encoder_length', 60),
                                                 max_prediction_length=params.get('max_prediction_length', 10))
                dl_try = ds_try.to_dataloader(train=False, batch_size=64, num_workers=0)
            except Exception:
                dl_try = None

            if dl_try is None:
                return False

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pred = candidate_model.predict(dl_try)

            if pred is None:
                return False
            return True
        except Exception:
            return False

    # ========================================
    # STRATEGY 1: PyTorch Lightning API (fast path)
    # ========================================
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint_path),
            map_location=map_location,
        )

        # Extract dataset params from model hparams
        dataset_params = {
            'max_encoder_length': getattr(model.hparams, 'max_encoder_length', 60),
            'max_prediction_length': getattr(model.hparams, 'max_prediction_length', 10),
        }

        # Validate the loaded Lightning model by performing a lightweight predict
        try:
            if _validate_model_predict(model, dataset_params):
                logger.info("Loaded via Lightning API and validated with a small predict")
                return model, dataset_params
            else:
                logger.info("Lightning model loaded but failed validation - will try stateless fallbacks")
        except Exception as e_val:
            logger.debug("Error while validating Lightning-loaded model: %s", e_val)
    except Exception as e:
        logger.debug("Lightning load attempt failed: %s", e)
    
    # ========================================
    # STRATEGY 2: Load Config JSON + Checkpoint (stateless constructor)
    # ========================================
    try:
        # Find config JSON - check new paths first, then legacy
        config_candidates = []
        if ModelPaths is not None and symbol:
            paths = ModelPaths(symbol)
            config_candidates.append(paths.tft.config)
        # Add legacy paths
        config_candidates.extend([
            Path('saved_models') / symbol / 'tft' / 'config.json' if symbol else None,
            Path('saved_models') / f'{symbol}_tft_config.json' if symbol else None,
            checkpoint_path.parent / 'config.json',
            checkpoint_path.parent.parent / f'{checkpoint_path.parent.name}_tft_config.json',
        ])
        
        config_path = None
        for candidate in config_candidates:
            if candidate and candidate.exists():
                config_path = candidate
                break
        
        if not config_path:
            raise FileNotFoundError(f"No TFT config JSON found. Searched: {config_candidates}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        logger.info("Loaded config: %s", config_path)
        
        # Extract TFT constructor params (FILTER OUT INVALID KEYS)
        valid_keys = {
            'hidden_size', 'attention_head_size', 'dropout',
            'hidden_continuous_size', 'output_size', 'learning_rate',
            'reduce_on_plateau_patience'
        }
        
        tft_kwargs = {k: v for k, v in config.items() if k in valid_keys}
        
        # Handle architecture mismatches
        if 'num_attention_heads' in config and 'attention_head_size' not in tft_kwargs:
            hidden_size = config.get('hidden_size', 128)
            num_heads = config.get('num_attention_heads', 4)
            tft_kwargs['attention_head_size'] = hidden_size // num_heads
            logger.debug("Computed attention_head_size = %s from num_heads=%s", tft_kwargs['attention_head_size'], num_heads)
        
        # Add loss function
        tft_kwargs['loss'] = QuantileLoss()
        
        # Load checkpoint state dict
        # prefer the original state dict we loaded earlier
        state_dict = original_state_dict if isinstance(original_state_dict, dict) else {}

        # Strip 'model.' prefix if present (redundant with stripped_state_dict but keep for clarity)
        if isinstance(state_dict, dict) and any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
            logger.debug("Stripped 'model.' prefix from %d keys", len(state_dict))
        
        # Create model instance (without dataset - fallback mode)
        logger.debug("Creating TFT with params: %s", list(tft_kwargs.keys()))
        model = TemporalFusionTransformer(**tft_kwargs)
        
        # Load state dict (quietly capture big dumps)
        missing, unexpected, err = load_state_dict_quietly(model, state_dict, strict=False)
        if err:
            logger.debug("Config+state load produced error: %s", err)
        if missing:
            logger.debug("Missing %d keys (first 5): %s", len(missing), list(missing)[:5])
        if unexpected:
            logger.debug("Unexpected %d keys (first 5): %s", len(unexpected), list(unexpected)[:5])
        
        # Extract dataset params from config
        dataset_params = {
            'max_encoder_length': config.get('max_encoder_length', 60),
            'max_prediction_length': config.get('max_prediction_length', 10),
        }
        
        logger.info("Loaded via config JSON + state_dict (best-effort)")
        return model, dataset_params
        
    except Exception as e:
        logger.debug("Strategy 2 failed: %s", e)
        import traceback
        traceback.print_exc()
    
    # ========================================
    # STRATEGY 3: Instantiate from dataset and apply conservative key-copy
    # ========================================
    try:
        # Build dataset (already attempted earlier via get_cached_tft_dataset above)
        dataset = get_cached_tft_dataset(symbol, max_encoder_length=60, max_prediction_length=10)

        # Instantiate a model that includes variable selection/prescalers matching current dataset
        tft_model = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=128,
            attention_head_size=max(1, 128 // 4),
            dropout=0.1,
            hidden_continuous_size=64,
            output_size=7,
            loss=QuantileLoss(),
        )

        # Try multiple load variants (original -> stripped -> remapped -> exact-shape)
        for variant_name, candidate in (
            ("original", original_state_dict),
            ("stripped", stripped_state_dict),
        ):
            if not isinstance(candidate, dict):
                continue
            missing, unexpected, err = load_state_dict_quietly(tft_model, candidate, strict=False)
            logger.debug("Variant '%s' load: missing=%d unexpected=%d err=%s", variant_name, len(missing), len(unexpected), err)
            if err is None and len(missing) <= max(1, int(len(candidate) * 0.5)):
                ds_params = {'max_encoder_length': 60, 'max_prediction_length': 10}
                # validate before returning
                if _validate_model_predict(tft_model, ds_params):
                    logger.info("Loaded TFT model using variant '%s'", variant_name)
                    return tft_model, ds_params
                logger.debug("Variant '%s' passed load but failed runtime prediction", variant_name)

        # Try adaptive remapping
        try:
            remapped, version_info = adapt_tft_checkpoint_keys(original_state_dict, tft_model)
            missing, unexpected, err = load_state_dict_quietly(tft_model, remapped, strict=False)
            logger.debug("Adaptive remap load: err=%s missing=%d", err, len(missing))
            if err is None:
                ds_params = {'max_encoder_length': 60, 'max_prediction_length': 10}
                logger.info("Loaded TFT model with adaptive remapping: %s", version_info)
                return tft_model, ds_params
        except Exception:
            logger.debug("Adaptive remapping failed; continuing to filtered exact-shape copy")

        # Final fallback: copy only keys whose shapes exactly match
        model_state = tft_model.state_dict()
        filtered = {}
        src_candidates = {}
        if isinstance(stripped_state_dict, dict) and stripped_state_dict:
            src_candidates = stripped_state_dict
        elif isinstance(original_state_dict, dict):
            src_candidates = original_state_dict

        for k, v in src_candidates.items():
            if k in model_state:
                try:
                    if tuple(model_state[k].shape) == tuple(v.shape):
                        filtered[k] = v
                except Exception:
                    continue

        if filtered:
            missing, unexpected, err = load_state_dict_quietly(tft_model, filtered, strict=False)
            logger.info("Partially loaded %d exact-shape keys into TFT model", len(filtered))
            ds_params = {'max_encoder_length': 60, 'max_prediction_length': 10}
            if _validate_model_predict(tft_model, ds_params):
                return tft_model, ds_params
            logger.debug("Conservative exact-shape load passed but model failed runtime prediction")

    except Exception as e:
        logger.error("Final from-dataset loading fallback failed: %s", e)

    logger.error("All TFT loading strategies failed")
    return None, {}

def prepare_inference_data(
    symbol: str,
    use_cache_only: bool = False,
    include_sentiment: bool = True  # ← ADD THIS PARAMETER
) -> pd.DataFrame:
    """Prepare data for TFT inference with mandatory caching for backtests.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        use_cache_only: If True, NEVER fetch data - only load from cache
        include_sentiment: Include sentiment features (default True)
    
    Returns:
        DataFrame with shape (n_days, 147 features) + time_idx column
    
    Raises:
        FileNotFoundError: If use_cache_only=True but no cache exists
        ValueError: If symbol appears to be a filename instead of ticker
    """
    from data.cache_manager import DataCacheManager
    from data.feature_engineer import get_feature_columns
    import re
    
    # CRITICAL: Validate symbol is a real ticker (not a filename)
    # Block: BEST_MODEL, BEST_MODEL-V9, TFT, etc.
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        raise ValueError(
            f"Invalid symbol '{symbol}' - must be 1-5 uppercase letters. "
            f"Detected filename-like pattern. Check TFT loader symbol inference."
        )
    
    cache_manager = DataCacheManager()
    
    if use_cache_only:
        # ========================================
        # BACKTEST MODE: CACHE ONLY (NO FETCH)
        # ========================================
        logger.info("CACHE-ONLY MODE: Loading %s (no network fetch allowed)", symbol)
        
        try:
            cached = cache_manager._load_cache(symbol)
            
            if cached is None:
                raise FileNotFoundError(
                    f"\n❌ No cached data for {symbol}\n"
                    f"   Cache required for backtest mode (use_cache_only=True)\n\n"
                    f"   FIX: Run this command ONCE to create cache:\n"
                    f"   python -c \"from data.cache_manager import DataCacheManager; "
                    f"cm = DataCacheManager(); "
                    f"cm.get_or_fetch_data('{symbol}', include_sentiment=True, force_refresh=True); "
                    f"print('✓ Cache created for {symbol}')\"\n"
                )
            
            # Extract engineered DataFrame from cache
            if isinstance(cached, dict) and 'engineered' in cached:
                df = cached['engineered'].copy()
            elif isinstance(cached, (tuple, list)) and len(cached) >= 2:
                df = cached[1].copy()  # (raw, engineered, prepared, cols)
            else:
                raise ValueError(f"Unexpected cache format for {symbol}: {type(cached)}")
            
            logger.info("Loaded %d rows from cache (no fetch)", len(df))
            
        except FileNotFoundError:
            raise  # Re-raise with instructions
        except Exception as e:
            raise RuntimeError(f"Failed to load cache for {symbol}: {e}")
    
    else:
        # ========================================
        # TRAINING MODE: ALLOW FETCH IF NEEDED
        # ========================================
        logger.info("TRAINING MODE: Fetching %s (cache will be updated)", symbol)
        
        _, df, _, _ = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=include_sentiment,
            force_refresh=False  # Use cache if valid
        )
    
    # ========================================
    # ADD time_idx COLUMN (REQUIRED BY TFT)
    # ========================================
    if 'time_idx' not in df.columns:
        # Reset datetime index to column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=False)
            if 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
        
        # Create integer time index
        df['time_idx'] = range(len(df))
        logger.info("Added time_idx column (0 to %d)", len(df)-1)
    
    # Validate time_idx
    if not pd.api.types.is_integer_dtype(df['time_idx']):
        df['time_idx'] = df['time_idx'].astype(int)
    
    if not df['time_idx'].is_monotonic_increasing:
        logger.warning("Sorting by time_idx (was not monotonic)")
        df = df.sort_values('time_idx').reset_index(drop=True)
    
    # ========================================
    # VALIDATE FEATURES
    # ========================================
    expected_features = get_feature_columns(include_sentiment=True)
    present_features = [c for c in df.columns if c in expected_features]
    
    if len(present_features) != 147:
        missing = set(expected_features) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required features: {list(missing)[:10]}...\n"
                f"Re-run feature engineering with include_sentiment=True"
            )
    
    logger.info("Validated %d features", len(present_features))
    
    # Return with time_idx + features (DataFrame ready for TimeSeriesDataSet)
    return df


def prepare_tft_data_for_inference(symbol: str = None, df: pd.DataFrame = None,
                                   train_dataset=None, max_encoder_length: int = None,
                                   max_prediction_length: int = None, include_sentiment: bool = True):
    """Prepare a DataFrame matching the exact TFT training format.

    Ensures:
      - All canonical engineered features present (fills missing with neutral values)
      - `time_idx` is a sequential integer index
      - `symbol` (group id) column exists and is filled
      - `returns` target column exists (1-day return)
      - Uses same encoder/prediction lengths if provided via `train_dataset`
      - Validates final shape matches EXPECTED_FEATURE_COUNT

    Returns prepared DataFrame ready for TimeSeriesDataSet construction.
    """
    # Lazy imports to avoid import-time overhead
    try:
        from data.cache_manager import DataCacheManager
        from data.feature_engineer import engineer_features, validate_and_fix_features, get_feature_columns, EXPECTED_FEATURE_COUNT
    except Exception:
        # Try package-style import for alternate execution contexts
        from python_ai_service.data.cache_manager import DataCacheManager  # type: ignore
        from python_ai_service.data.feature_engineer import engineer_features, validate_and_fix_features, get_feature_columns, EXPECTED_FEATURE_COUNT  # type: ignore

    # Obtain data if only symbol provided
    if df is None:
        if symbol is None:
            raise ValueError('Either symbol or df must be provided')
        cm = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cm.get_or_fetch_data(symbol, include_sentiment=include_sentiment, force_refresh=False)
        # prefer the fully engineered dataframe (prepared_df may include target shifts)
        df = engineered_df if engineered_df is not None else raw_df

    df = df.copy()

    # Ensure required OHLCV columns exist for feature engineering
    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_ohlcv = [c for c in required_ohlcv if c not in df.columns]
    if missing_ohlcv:
        raise ValueError(f"Missing required OHLCV columns for feature engineering: {missing_ohlcv}")

    # Run feature engineering to produce canonical features (idempotent)
    try:
        df = engineer_features(df, symbol=symbol if symbol is not None else None, include_sentiment=include_sentiment)
    except Exception as e:
        # If feature engineering fails, attempt to continue with existing engineered columns
        logger.warning("engineer_features failed: %s. Proceeding with existing columns.", e)

    # Ensure `returns` target exists (1-day return)
    if 'returns' not in df.columns:
        df['returns'] = df['Close'].pct_change().fillna(0.0)

    # Ensure time_idx sequential integer
    if 'time_idx' not in df.columns:
        df = df.reset_index(drop=True)
        df['time_idx'] = np.arange(len(df))
    else:
        # Coerce to integer index if possible
        try:
            df['time_idx'] = df['time_idx'].astype(int)
        except Exception:
            df['time_idx'] = np.arange(len(df))

    # Ensure symbol/group id column exists (training used 'symbol')
    if 'symbol' not in df.columns:
        if symbol is None:
            raise ValueError('symbol must be provided when DataFrame has no group id column')
        df['symbol'] = symbol
    else:
        # Coerce to string ticker upper-case
        df['symbol'] = df['symbol'].astype(str).str.upper()

    # Reconcile canonical feature set (adds missing, drops extras, orders columns)
    try:
        df = validate_and_fix_features(df)
    except Exception as e:
        # validate_and_fix_features is defensive but may raise if severe mismatch
        logger.warning("validate_and_fix_features failed: %s. Falling back to get_feature_columns reconciliation.", e)
        canonical = get_feature_columns(include_sentiment=include_sentiment)
        # Ensure target and group+time are present
        for col in canonical:
            if col not in df.columns:
                df[col] = 0.0

    # Enforce types: numeric features to float32 for model efficiency
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        try:
            df[c] = df[c].astype(np.float32)
        except Exception:
            pass

    # Use encoder/prediction lengths from train_dataset if provided
    if train_dataset is not None:
        try:
            max_encoder_length = getattr(train_dataset, 'max_encoder_length', max_encoder_length)
            max_prediction_length = getattr(train_dataset, 'max_prediction_length', max_prediction_length)
        except Exception:
            pass

    # Validate dataset dimensions
    # Check feature count equals canonical expected
    feature_cols = [c for c in df.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol', 'time_idx']]
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise AssertionError(f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, got {len(feature_cols)}. Columns: {feature_cols[:10]}...")

    # Ensure we have at least max_encoder_length rows
    if max_encoder_length is not None and len(df) < max_encoder_length:
        raise ValueError(f"Not enough rows for encoder length: need {max_encoder_length}, got {len(df)}")

    # Final ordering: ensure time_idx and symbol columns exist and features follow
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ordered = [c for c in ['time_idx', 'symbol'] if c in df.columns] + [c for c in df.columns if c not in ['time_idx', 'symbol'] + exclude_cols]
    try:
        df = df.reindex(columns=ordered)
    except Exception:
        pass

    logger.info("Prepared TFT inference DataFrame for symbol=%s: shape=%s, features=%d", symbol, df.shape, len(feature_cols))
    return df
def generate_forecast(model, df, horizon=5):
    """
    Generate multi-horizon forecast using trained TFT model.
    
    Args:
        model: Trained TemporalFusionTransformer
        df: DataFrame with time_idx + features (from prepare_inference_data)
        horizon: Number of steps to forecast
    
    Returns:
        Dict with keys: ['horizons', 'predictions']
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    import torch
    import numpy as np
    
    print(f"Generating {horizon}-step forecast...")
    
    # ========================================
    # STEP 1: Extract model's expected features from hparams
    # ========================================
    try:
        # Get the feature lists the model was trained with
        if hasattr(model, 'hparams'):
            hparams = model.hparams
            
            # Extract feature columns from model config
            time_varying_unknown_reals = getattr(hparams, 'time_varying_unknown_reals', [])
            time_varying_known_reals = getattr(hparams, 'time_varying_known_reals', [])
            time_varying_known_categoricals = getattr(hparams, 'time_varying_known_categoricals', [])
            static_categoricals = getattr(hparams, 'static_categoricals', [])
            static_reals = getattr(hparams, 'static_reals', [])
            target = getattr(hparams, 'target', 'returns')
            max_encoder_length = getattr(hparams, 'max_encoder_length', 60)
            max_prediction_length = getattr(hparams, 'max_prediction_length', 10)
            
            print(f"  Model expects {len(time_varying_unknown_reals)} unknown reals")
            print(f"  Target: {target}")
            print(f"  Encoder length: {max_encoder_length}")
            
        else:
            raise ValueError("Model has no hparams - cannot reconstruct dataset")
    
    except Exception as e:
        print(f"❌ Failed to extract model hyperparameters: {e}")
        raise
    
    # ========================================
    # STEP 2: Prepare DataFrame with required columns
    # ========================================
    df_forecast = df.copy()
    
    # Ensure required columns exist
    required_cols = ['time_idx', target]
    missing = [c for c in required_cols if c not in df_forecast.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add symbol if not present
    if 'symbol' not in df_forecast.columns:
        df_forecast['symbol'] = 'AAPL'
    
    # Add any missing categorical columns with defaults
    for cat in static_categoricals + time_varying_known_categoricals:
        if cat not in df_forecast.columns:
            df_forecast[cat] = 'unknown'
    
    # Add any missing real columns with zeros
    for real in static_reals + time_varying_known_reals:
        if real not in df_forecast.columns:
            df_forecast[real] = 0.0
    
    # CRITICAL: Filter to only columns the model expects
    # Remove any extra columns that weren't in training
    expected_cols = set(
        ['time_idx', 'symbol', target] +
        list(time_varying_unknown_reals) +
        list(time_varying_known_reals) +
        list(time_varying_known_categoricals) +
        list(static_categoricals) +
        list(static_reals)
    )
    
    # Keep Date column if present (for reference)
    if 'Date' in df_forecast.columns:
        expected_cols.add('Date')
    
    # Drop unexpected columns
    extra_cols = [c for c in df_forecast.columns if c not in expected_cols]
    if extra_cols:
        print(f"  Dropping {len(extra_cols)} extra columns not in training")
        df_forecast = df_forecast[[c for c in df_forecast.columns if c in expected_cols]]
    
    print(f"  DataFrame shape after filtering: {df_forecast.shape}")
    
    # ========================================
    # STEP 3: Create TimeSeriesDataSet (matching training)
    # ========================================
    try:
        dataset = TimeSeriesDataSet(
            df_forecast,
            time_idx='time_idx',
            target=target,
            group_ids=['symbol'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=min(horizon, max_prediction_length),
            min_encoder_length=max_encoder_length // 2,  # Allow shorter sequences
            min_prediction_length=1,
            
            # Use exact same features as training
            static_categoricals=list(static_categoricals),
            static_reals=list(static_reals),
            time_varying_known_categoricals=list(time_varying_known_categoricals),
            time_varying_known_reals=list(time_varying_known_reals),
            time_varying_unknown_reals=list(time_varying_unknown_reals),
            
            # Match training dataset settings
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            
            # Use GroupNormalizer (same as training)
            target_normalizer=GroupNormalizer(
                groups=['symbol'],
                transformation='softplus'
            ),
        )
        
        print(f"  ✓ Created dataset with {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Failed to create TimeSeriesDataSet: {e}")
        raise
    
    # ========================================
    # STEP 4: Create DataLoader
    # ========================================
    try:
        dataloader = dataset.to_dataloader(
            train=False,
            batch_size=128,
            num_workers=0  # Windows compatibility
        )
        
        print(f"  ✓ Created dataloader with {len(dataloader)} batches")
        
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        raise
    
    # ========================================
    # STEP 5: Generate Predictions
    # ========================================
    try:
        # Ensure the loss has a valid encoder to avoid NoneType callables
        try:
            ensure_loss_encoder(model)
        except Exception:
            pass
        model.eval()
        
        # Predict (keep call minimal to avoid forwarding unexpected kwargs into forward)
        try:
            raw_predictions = model.predict(dataloader)
        except TypeError:
            # Older/newer PF versions may differ in signature; try a more explicit call
            raw_predictions = model.predict(dataloader, mode='prediction')
        
        if raw_predictions is None:
            raise RuntimeError(
                "model.predict() returned None. This usually means:\n"
                "1. Dataset format doesn't match training\n"
                "2. Model expects different columns\n"
                "3. TimeSeriesDataSet creation failed silently\n\n"
                "Debug: Check that df columns match model.hparams features"
            )
        
        print(f"  ✓ Predictions shape: {raw_predictions.shape}")
        
        # Convert to numpy
        if torch.is_tensor(raw_predictions):
            predictions = raw_predictions.cpu().numpy()
        else:
            predictions = np.array(raw_predictions)
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ========================================
    # STEP 6: Format Output
    # ========================================
    # predictions shape: (samples, horizons, quantiles)
    # For TFT with QuantileLoss, typically 7 quantiles:
    # [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    
    if predictions.ndim == 3:
        num_samples, num_horizons, num_quantiles = predictions.shape
        print(f"  Quantile predictions: {num_quantiles} quantiles × {num_horizons} horizons")
        
        # Extract last sample (most recent)
        last_sample = predictions[-1]  # Shape: (horizons, quantiles)
        
        # Format as dict
        forecast = {
            'horizons': list(range(1, num_horizons + 1)),
            'predictions': {}
        }
        
        # Map quantile indices to values
        quantile_map = {
            0: 'q02',
            1: 'q10', 
            2: 'q25',
            3: 'q50',  # Median
            4: 'q75',
            5: 'q90',
            6: 'q98'
        }
        
        for h_idx in range(num_horizons):
            h = h_idx + 1
            forecast['predictions'][h] = {}
            
            for q_idx, q_name in quantile_map.items():
                if q_idx < num_quantiles:
                    forecast['predictions'][h][q_name] = float(last_sample[h_idx, q_idx])
        
    else:
        # Fallback for unexpected shape
        print(f"  ⚠️ Unexpected prediction shape: {predictions.shape}")
        forecast = {
            'horizons': list(range(1, min(horizon, predictions.shape[1]) + 1)),
            'predictions': {
                h: {'q50': float(predictions[-1, h-1])}
                for h in range(1, min(horizon, predictions.shape[1]) + 1)
            }
        }
    
    print(f"  ✓ Forecast generated for {len(forecast['predictions'])} horizons")
    
    return forecast

def generate_trading_signal(forecast, threshold=0.01):
    """
    Simple logic to generate a signal from forecast.
    """
    # Use horizon 1 median
    try:
        h1_pred = forecast['predictions'][1]['q50']
        if h1_pred > threshold:
            return 1 # BUY
        elif h1_pred < -threshold:
            return -1 # SELL
        else:
            return 0
    except KeyError:
        return 0


def inspect_tft_checkpoint(checkpoint_path, max_sample=30):
    """Quick diagnostic: summarize a TFT checkpoint without dumping everything.

    Prints top-level keys, presence of '__special_save__'/'hyper_parameters',
    number of state_dict entries, whether keys use 'model.' prefix, and a
    short sampled listing with shapes.
    """
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        logger.error("Failed to read checkpoint: %s", e)
        return

    logger.info("Checkpoint type: %s", type(ckpt))
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        logger.info("Top-level keys (%d): %s", len(keys), keys)
        if '__special_save__' in ckpt:
            logger.info("Contains '__special_save__'")
        if 'hyper_parameters' in ckpt:
            logger.info("Contains 'hyper_parameters'")
        # Extract state dict
        state = ckpt.get('state_dict', ckpt)
    else:
        print("Not a dict checkpoint; cannot inspect state_dict keys reliably")
        return

    if not isinstance(state, dict):
        logger.error("State is not a dict; aborting")
        return

    total = len(state)
    logger.info("State dict keys: %d", total)
    sample = list(state.keys())[:max_sample]
    has_model_prefix = any(k.startswith('model.') for k in state.keys())
    has_module_prefix = any(k.startswith('module.') for k in state.keys())
    logger.info("Has 'model.' prefix: %s; Has 'module.' prefix: %s", has_model_prefix, has_module_prefix)

    logger.info("Sample (%d) keys and shapes:", len(sample))
    for k in sample:
        try:
            v = state[k]
            shape = getattr(v, 'shape', None)
            logger.info("  %s -> shape=%s", k, shape)
        except Exception:
            logger.info("  %s -> <unreadable>", k)

    if total > max_sample:
        logger.info("... (%d more keys)", total - max_sample)

    logger.info("Diagnostic complete.")


def extract_symbol_from_checkpoint_path(path: str) -> str:
        """Extract a ticker symbol from a checkpoint path.

        Rules:
        - If the path contains a folder named 'tft' (case-insensitive), the symbol is
            taken as the next path component after 'tft'.
        - Otherwise, the function will inspect nearby path components (stem, parent,
            grandparent) for a candidate.
        - The candidate is uppercased and validated against the strict ticker pattern
            `[A-Z]{1,5}`. If it does not match, a ValueError is raised.

        This avoids accidentally treating folder names like 'tft' or filenames like
        'best_model-v9' as tickers.
        """
        from pathlib import Path
        import re

        p = Path(path)
        parts = [str(part) for part in p.parts if str(part)]
        lower = [part.lower() for part in parts]

        # Prefer the component immediately following a 'tft' folder
        if 'tft' in lower:
                idx = lower.index('tft')
                if idx + 1 < len(parts):
                        candidate = parts[idx + 1].upper()
                        if re.fullmatch(r"[A-Z]{1,5}", candidate):
                                return candidate
                        raise ValueError(f"Inferred candidate '{candidate}' after 'tft' is not a valid ticker (expected [A-Z]{{1,5}})")
                raise ValueError(f"Checkpoint path contains 'tft' but no following component to infer symbol: {path}")

        # Fallback: inspect likely nearby components (stem, parent, grandparent)
        candidates = [p.stem, p.parent.name, p.parent.parent.name, p.parent.parent.parent.name]
        for c in candidates:
                if not c:
                        continue
                cand = str(c).upper()
                if re.fullmatch(r"[A-Z]{1,5}", cand):
                        return cand

        raise ValueError(f"Could not extract valid ticker symbol from checkpoint path: {path}. "
                                         "Expected pattern [A-Z]{1,5} in path components (e.g., saved_models/tft/AAPL/best_model.ckpt)")

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

logger = logging.getLogger(__name__)


def extract_dataset_params(model: Any) -> Dict[str, Any]:
    """Extract dataset parameters from loaded TFT model.

    This helper tries several locations where older/newer Lightning/PyTorch-Forecasting
    packages may store hyperparameters. It returns a conservative dict of dataset
    parameters that other code can use to rebuild or validate datasets.
    """
    try:
        h = getattr(model, "hparams", None) or getattr(model, "hyper_parameters", None) or {}
        # hparams may be a Namespace-like, dict-like or attribute container
        def _get(k, default=None):
            if not h:
                return default
            try:
                if isinstance(h, dict):
                    return h.get(k, default)
                return getattr(h, k, default)
            except Exception:
                return default

        return {
            "max_encoder_length": _get("max_encoder_length", 60),
            "max_prediction_length": _get("max_prediction_length", 10),
            "time_varying_known_reals": _get("time_varying_known_reals", []),
            "time_varying_unknown_reals": _get("time_varying_unknown_reals", []),
            "static_categoricals": _get("static_categoricals", []),
        }
    except KeyError as e:
        if "time_idx" in str(e):
            logger.warning("KeyError 'time_idx' when extracting dataset params; returning empty dataset params")
            return {}
        raise
    except Exception:
        return {}


def _find_state_dict_in_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    # Common keys across PL versions
    possible = [
        "state_dict",
        "model_state_dict",
        "model",
        "state_dicts",
        "checkpoint",
    ]
    for k in possible:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]

    # Sometimes the checkpoint itself IS a state_dict (mapping tensor names to tensors)
    # Detect by checking value types
    if all(isinstance(v, torch.Tensor) or hasattr(v, "shape") for v in ckpt.values()):
        return ckpt

    # Nothing found
    return {}


def load_tft_model(checkpoint_path: str, map_location: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """Load TFT model with fallback strategies for checkpoint compatibility.

    Returns (model, dataset_params) on success or (None, {}) on failure.

    Strategies (in order):
    - Use TemporalFusionTransformer.load_from_checkpoint (Lightning 2.x / 1.x usual API)
    - If that fails, attempt pl.LightningModule.load_from_checkpoint variant
    - If still failing, load raw checkpoint with torch.load, reconstruct model, and load state_dict

    The loader handles several common mismatches: prefixed keys ("model.", "module."),
    missing/extra keys (loads with strict=False), mapping from legacy hyperparameters
    (e.g. `num_attention_heads`) to expected ones, and missing dataset params.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error("Checkpoint path does not exist: %s", checkpoint_path)
        return None, {}

    # Strategy 1: try the standard class method loader (preferred)
    try:
        try:
            model = TemporalFusionTransformer.load_from_checkpoint(
                str(checkpoint_path), map_location=map_location
            )
        except AttributeError as e:
            # Some Lightning versions raise weird AttributeErrors referencing internal
            # save mechanisms (e.g. '__special_save__'); try using the pl LightningModule
            # generic loader as a fallback.
            if "__special_save__" in str(e):
                logger.info("Caught special save AttributeError; trying pl.LightningModule.load_from_checkpoint")
                model = pl.LightningModule.load_from_checkpoint(str(checkpoint_path), map_location=map_location)
            else:
                raise

        dataset_params = extract_dataset_params(model)
        return model, dataset_params
    except Exception as e:
        logger.warning("Lightning class loader failed: %s", e)

    # Strategy 2: manual torch.load + reconstruct
    try:
        ckpt = torch.load(str(checkpoint_path), map_location=map_location)

        # Extract hparams from common locations used by PL and others
        hparams = {}
        for key in ("hyper_parameters", "hyper_params", "hparams", "hparams_saved", "args"):
            if isinstance(ckpt.get(key), dict):
                hparams = ckpt.get(key)
                break
            if hasattr(ckpt.get(key), "__dict__"):
                try:
                    hparams = vars(ckpt.get(key))
                    break
                except Exception:
                    pass

        # Also check inside ckpt.get('checkpoint')
        if not hparams:
            nested = ckpt.get("checkpoint", {}) or {}
            if isinstance(nested, dict) and nested.get("hyper_parameters"):
                hparams = nested.get("hyper_parameters", {})

        # Prepare TFT kwargs with safe defaults and apply mappings for legacy keys
        def _get_h(k, default=None):
            try:
                if isinstance(hparams, dict):
                    return hparams.get(k, default)
                return getattr(hparams, k, default)
            except Exception:
                return default

        tft_kwargs: Dict[str, Any] = {
            "hidden_size": _get_h("hidden_size", 128),
            "attention_head_size": _get_h("attention_head_size", 4),
            "dropout": _get_h("dropout", 0.1),
            "hidden_continuous_size": _get_h("hidden_continuous_size", 8),
            "output_size": _get_h("output_size", 7),
            "loss": _get_h("loss", QuantileLoss()),
            "learning_rate": _get_h("learning_rate", 1e-3),
        }

        # Handle legacy num_attention_heads -> attention_head_size mapping
        if (_get_h("num_attention_heads") is not None) and (_get_h("attention_head_size") is None):
            try:
                num_heads = int(_get_h("num_attention_heads"))
                hidden = int(tft_kwargs.get("hidden_size", 128))
                if num_heads > 0:
                    tft_kwargs["attention_head_size"] = max(1, hidden // num_heads)
            except Exception:
                pass

        # Try to find dataset params inside checkpoint
        dataset_params = {}
        for key in ("dataset_parameters", "dataset_params", "data_params"):
            v = ckpt.get(key)
            if isinstance(v, dict):
                dataset_params = v
                break

        # Fallback to config JSON sitting beside checkpoint (common pattern)
        if not dataset_params:
            config_path = checkpoint_path.parent / f"{checkpoint_path.stem.split('-')[0]}_tft_config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        dataset_params = json.load(f)
                except Exception as e:
                    logger.warning("Failed to read config JSON %s: %s", config_path, e)

        # Build model skeleton
        try:
            if dataset_params:
                # The user may pass a dataset object in some saved checkpoints; try to pass-through
                model = TemporalFusionTransformer.from_dataset(dataset_params, **tft_kwargs)
            else:
                # Last resort: create model without dataset (may be sufficient for loading weights)
                logger.warning("Creating TemporalFusionTransformer without dataset params")
                model = TemporalFusionTransformer(**tft_kwargs)
        except KeyError as e:
            # Some older checkpoints may require dataset validation using 'time_idx'
            if "time_idx" in str(e):
                logger.warning("Missing 'time_idx' during model construction; proceeding without dataset validation")
                model = TemporalFusionTransformer(**tft_kwargs)
            else:
                raise

        # Extract state_dict from checkpoint (multiple possible layouts)
        state_dict = _find_state_dict_in_checkpoint(ckpt)

        # If still empty, try common nested locations
        if not state_dict:
            for candidate in ("state_dict", "model_state_dict", "weights", "params"):
                v = ckpt.get(candidate)
                if isinstance(v, dict):
                    state_dict = v
                    break

        if not state_dict:
            logger.warning("No state_dict found in checkpoint; returning model without weights")
            return model, dataset_params or {}

        # Normalize keys: strip common prefixes
        def _normalize_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
            new = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("model."):
                    nk = nk[len("model."):]
                if nk.startswith("module."):
                    nk = nk[len("module."):]
                # some checkpoints have "net." prefix
                if nk.startswith("net."):
                    nk = nk[len("net."):]
                new[nk] = v
            return new

        if any(k.startswith("model.") or k.startswith("module.") or k.startswith("net.") for k in state_dict.keys()):
            state_dict = _normalize_keys(state_dict)

        # Try loading weights; use strict=False to tolerate mismatch
        try:
            load_result = model.load_state_dict(state_dict, strict=False)
            # load_result may be a NamedTuple with missing_keys/unexpected_keys or a plain tuple
            missing = getattr(load_result, "missing_keys", None) or (load_result[0] if isinstance(load_result, (list, tuple)) and len(load_result) > 0 else [])
            unexpected = getattr(load_result, "unexpected_keys", None) or (load_result[1] if isinstance(load_result, (list, tuple)) and len(load_result) > 1 else [])
            if missing:
                logger.warning("Missing keys when loading state_dict: %s", missing[:10])
            if unexpected:
                logger.warning("Unexpected keys when loading state_dict: %s", unexpected[:10])
        except RuntimeError as e:
            logger.warning("RuntimeError loading state_dict (trying non-strict): %s", e)
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e2:
                logger.error("Failed to load state_dict even with strict=False: %s", e2)
                return None, {}

        return model, dataset_params or {}

    except Exception as e:
        # Handle a few specific error messages requested by the user
        msg = str(e)
        if "num_attention_heads" in msg and "TypeError" in type(e).__name__:
            logger.warning("TypeError referencing num_attention_heads encountered: %s", e)
        if "time_idx" in msg or isinstance(e, KeyError):
            logger.warning("KeyError 'time_idx' encountered during load; skipping dataset validation")
            return None, {}

        logger.exception("Manual checkpoint load failed: %s", e)
        return None, {}

"""Shared utilities.

Keep TensorFlow-dependent imports optional so GBM-only tooling can run in a
lighter environment.
"""

__all__ = []

try:
    from utils.model_io import get_model_info, load_model_artifacts, load_model_safe, save_model_with_metadata

    __all__.extend(
        [
            "load_model_safe",
            "load_model_artifacts",
            "save_model_with_metadata",
            "get_model_info",
        ]
    )
except Exception:
    pass

try:
    from utils.losses import directional_mse, focal_loss, get_custom_objects, register_custom_objects

    __all__.extend(
        [
            "directional_mse",
            "focal_loss",
            "get_custom_objects",
            "register_custom_objects",
        ]
    )
except Exception:
    pass

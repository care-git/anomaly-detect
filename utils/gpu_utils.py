# utils/gpu_utils.py

import gc
import os
import sys

# Suppress TensorFlow C++ INFO and WARNING messages before TF is imported.
# These bypass Python logging and print with a different format, polluting
# the terminal output. Level 2 = suppress INFO + WARNING, keep ERROR + FATAL.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from utils.logger import get_logger

logger = get_logger(__name__, "INFO")

try:
    import cuml  # noqa: F401
    _CUML_AVAILABLE = True
except ImportError:
    _CUML_AVAILABLE = False

_GPU_CONFIGURED = False


def cuml_available() -> bool:
    """Returns True if RAPIDS cuML is installed and importable."""
    return _CUML_AVAILABLE


def setup_gpu() -> None:
    """Configures TF GPU memory growth and logs available compute devices.

    Idempotent - safe to call from multiple sites; configuration and logging
    only run once per process. Must be called before any TF GPU operation.
    """
    global _GPU_CONFIGURED
    if _GPU_CONFIGURED:
        return

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            if sys.platform == "win32":
                logger.info(
                    "TensorFlow GPU devices: %s (DirectML - DirectX 12 acceleration).",
                    [g.name for g in gpus],
                )
            else:
                logger.info("TensorFlow GPU devices: %s", [g.name for g in gpus])
        else:
            logger.info("TensorFlow: no GPU found - training on CPU.")
    except Exception:
        pass

    if _CUML_AVAILABLE:
        logger.info("cuML (RAPIDS) available - GPU-accelerated RF/SVM enabled.")
    else:
        if sys.platform == "win32":
            logger.info(
                "cuML not available - RF/SVM will use sklearn (CPU). "
                "cuML requires Linux or WSL2; see README for the WSL2 installation path."
            )
        else:
            logger.info(
                "cuML not available - RF/SVM will use sklearn (CPU). "
                "Install RAPIDS via conda to enable GPU support."
            )

    _GPU_CONFIGURED = True


def release_gpu_memory() -> None:
    """Releases TF GPU memory and runs garbage collection.

    Safe to call after any model - is a no-op when TF has not been used.
    Intended for use in benchmark and training orchestrators between model runs.
    """
    try:
        import keras
        keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

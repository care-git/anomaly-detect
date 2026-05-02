# utils/gpu_utils.py

import os

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


def cuml_available() -> bool:
    """Returns True if RAPIDS cuML is installed and importable."""
    return _CUML_AVAILABLE


def log_gpu_info() -> None:
    """Logs available compute devices visible to TensorFlow and cuML."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info("TensorFlow GPU devices: %s", [g.name for g in gpus])
        else:
            logger.info("TensorFlow: no GPU found — training on CPU.")
    except Exception:
        pass

    if _CUML_AVAILABLE:
        logger.info("cuML (RAPIDS) available — GPU-accelerated RF/SVM enabled.")
    else:
        logger.info("cuML not available — RF/SVM will use sklearn (CPU). "
                    "Install RAPIDS via conda to enable GPU support.")

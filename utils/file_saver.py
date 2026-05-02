# utils/file_saver.py

import os
import glob
import joblib
import json
from datetime import datetime
from scapy.all import wrpcap

from utils.logger import get_logger

logger = get_logger(__name__, "INFO")


def ensure_dir(path):
    """
    Ensures the directory for a given file or folder path exists.

    Parameters:
        path (str): File or Directory path.
    """
    dir_path = path if os.path.isdir(path) else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.debug("Created directory: %s", dir_path)


def save_pcap(pcap, path):
    """
    Saves a list of packets to a PCAP file.
    
    Parameters:
        pcap (list): List of Scapy packets.
        path (str): Destination file path.
    """
    ensure_dir(path)
    wrpcap(path, pcap)
    logger.info("PCAP saved: %s", path)


def save_pickle(obj, path):
    """
    Saves a Python object using joblib.

    Args:
        obj: Object to serialise.
        path (str): Destination file path.
    """
    ensure_dir(path)
    joblib.dump(obj, path)
    logger.info("Pickle saved: %s", path)


def save_json(obj, path):
    """
    Saves a dictionary or list as a JSON file.
    
    Parameters:
        obj: Data to save.
        path (str): Destination file path.
    """
    ensure_dir(path)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    logger.info("JSON saved: %s", path)


def save_text(text, path):
    """
    Saves a string or text block to a plain text file.
    
    Parameters:
        text (str): Text content to write.
        path (str): Destination file path."""
    ensure_dir(path)
    with open(path, 'w') as f:
        f.write(str(text))
    logger.info("Text saved: %s", path)


def save_dataframe(df, path):
    """
    Saves a pandas DataFrame to a CSV file.
    
    Parameters:
        df (pd.DataFrame): Data to save.
        path (str): Destination file path.
    """
    ensure_dir(path)
    df.to_csv(path, index=False)
    logger.info("CSV saved: %s", path)


def save_keras_model(model, path):
    """
    Saves a compiled Keras model to disk.

    Parameters:
        model: Keras model object.
        path (str): Destination file path.
    """
    from tensorflow.keras.models import save_model
    
    ensure_dir(path)
    save_model(model, path)
    logger.info("Keras model saved: %s", path)


def generate_incremented_path(base_path, extension=None):
    """
    Generates a unique, timestamped output file path.

    Parameters:
        base_path (str): Original file path (either from cli input or `config.yml`).
        extension (str, optional): Optional override file extension.

    Returns:
        str: A file path with timestamp and increment suffix.
    """
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    ext = extension or os.path.splitext(base_path)[-1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pattern = os.path.join(base_dir, f"{base_name}_{timestamp}_*{ext}")
    existing = glob.glob(pattern)
    next_id = len(existing) + 1

    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{base_name}_{timestamp}_{next_id}{ext}")


def safe_save_path(base_path, extension=None):
    """
    Ensures file is saved safely. If file already exists, appends timestamped increment.

    Parameters:
        base_path (str): Original file path.
        extension (str, optional): Optional override file extension.

    Returns:
        str: Modified file path if original already exists, if not, returns original file path.
    """
    if os.path.exists(base_path):
        logger.warning("[!] File already exists: %s. Generating new path...", base_path)
        return generate_incremented_path(base_path, extension)
    return base_path


def get_base_filename(path):
    """
    Extracts the base filename from a given file path, without its extension.

    Parameters:
        path (str): Full file path.

    Returns:
        str: Filename without directory or extension.
    """
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]
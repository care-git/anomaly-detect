# cli/main.py

# Must be set before TensorFlow/Keras is imported anywhere in the process.
# Suppress TensorFlow C++ INFO and WARNING messages before TF is imported.
# These bypass Python logging and print with a different format and look bad.
# Level 2 = suppress INFO + WARNING, keep ERROR + FATAL.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0") 

# Suppress TF and absl Python-side loggers before any TF import.
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Suppress Scapy runtime warnings
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# Suppress deprecation warnings from Scapy-imported cryptography library
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Suppress Pandas FutureWarning about silent downcasting (fillna) - pandas 2.x only
warnings.simplefilter(action="ignore", category=FutureWarning)

# Suppress StandardScaler feature warning for autoencoders on live data
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

import argparse

from __version__ import __version__
from utils.config_loader import set_config_path
from core.capture import run_capture
from core.preprocessor import run_preprocessor
from core.dataset_utils import run_dataset_utils
from models.trainer import run_train_model
from models.detector import run_detection
from models.benchmark import run_benchmark


def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=f"Anomaly-Based Threat Detection CLI (v{__version__})"
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config file (default: config/config.yml)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"anomaly-detect v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Capture
    capture_parser = subparsers.add_parser("capture", help="Capture network traffic")
    capture_parser.add_argument("--live", action="store_true", help="Enable live packet monitoring")
    capture_parser.add_argument("--interface", help="Network interface to capture/monitor with")
    capture_parser.add_argument("--duration", type=int, help="Capture duration (seconds)")
    capture_parser.add_argument("--packet-count", type=int, help="Max packets to capture")
    capture_parser.add_argument("--output", help="Output path to save PCAP file (incompatible with live monitoring)")

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Convert PCAP to CSV features")
    preprocess_parser.add_argument("--input", help="Input PCAP filepath for preprocessing")
    preprocess_parser.add_argument("--label", type=int, help="Optional integer label to assign to all extracted packets (0 = normal, 1 = anomaly)")
    preprocess_parser.add_argument("--output", help="Output path to save CSV file")

    # Dataset Utils
    dataset_parser = subparsers.add_parser("dataset", help="Dataset management and transformation utilities")
    dataset_parser.add_argument("--combine", nargs='+', help="List CSVs or folder paths to combine")
    dataset_parser.add_argument("--balance", action="store_true", help="Balance labels after combining")
    dataset_parser.add_argument("--split", action="store_true", help="Split dataset into train/test")
    dataset_parser.add_argument("--output", help="Output path to save CSV file(s)")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a machine learning model")
    train_parser.add_argument("--model", help="Model type (autoencoder, random_forest, svm)")
    train_parser.add_argument("--input", help="Input CSV filepath for training")
    train_parser.add_argument("--output", help="Output directory to save model")
    train_parser.add_argument("--cv", action="store_true", help="Run k-fold cross-validation instead of a single train/evaluate pass")
    train_parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for cross-validation (default: 5)")

    # Detect
    detect_parser = subparsers.add_parser("detect", help="Run detection using a trained model")
    detect_parser.add_argument("--model", help="Model type (autoencoder, random_forest, svm)")
    detect_parser.add_argument("--model-path", help="Path to model directory")
    detect_parser.add_argument("--live", action="store_true", help="Enable live packet detection")
    detect_parser.add_argument("--interface", help="Network interface to monitor with")
    detect_parser.add_argument("--input", help="Input PCAP filepath for detection (incompatible with live detection)")
    detect_parser.add_argument("--output", help="Output path to save prediction CSV file (incompatible with live detection)")

    # Benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Compare all models on the same labelled dataset")
    benchmark_parser.add_argument("--input", required=True, help="Input labelled CSV filepath")
    benchmark_parser.add_argument("--output", help="Directory to save benchmark results")

    return parser.parse_args()

def main():
    """
    Main CLI entry point.

    Routes parsed command-line arguments to their related module dispatcher.
    """
    args = parse_args()

    # Apply custom config path before any dispatcher loads config values.
    set_config_path(args.config)

    if args.command == "capture":
        run_capture(args)

    elif args.command == "preprocess":
        run_preprocessor(args)

    elif args.command == "dataset":
        run_dataset_utils(args)

    elif args.command == "train":
        run_train_model(args)

    elif args.command == "detect":
        run_detection(args)

    elif args.command == "benchmark":
        run_benchmark(args)
            

if __name__ == "__main__":
    main()

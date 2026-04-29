# --- This block can be removed without affecting pipeline in any way ---

# Suppress Scapy runtime warnings
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# Suppress deprecation warnings from Scapy-imported cryptography library
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Suppress Pandas deprecation warning about silent downcasting (fillna)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Suppress Tensorflow/Keras suggestion to swap to .legacy.Adam optimiser on M1/M2 chips
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress StandardScaler feature warning for autoencoders on live data
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# --- End of warning suppression block ---

# cli/main.py

import argparse

from __version__ import __version__
from core.capture import run_capture
from core.preprocessor import run_preprocessor
from core.dataset_utils import run_dataset_utils
from models.trainer import run_train_model
from models.detector import run_detection


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
    preprocess_parser.add_argument("--label", help="Optional label to assign to all extracted packets")
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

    # Detect
    detect_parser = subparsers.add_parser("detect", help="Run detection using a trained model")
    detect_parser.add_argument("--model", help="Model type (autoencoder, random_forest, svm)")
    detect_parser.add_argument("--model-path", help="Path to model directory")
    detect_parser.add_argument("--live", action="store_true", help="Enable live packet detection")
    detect_parser.add_argument("--interface", help="Network interface to monitor with")
    detect_parser.add_argument("--input", help="Input PCAP filepath for detection (incompatible with live detection)")
    detect_parser.add_argument("--output", help="Output path to save prediction CSV file (incompatible with live detection)")

    return parser.parse_args()

def main():
    """
    Main CLI entry point.
    
    Routes parsed command-line arguments to their related module dispatcher.
    """
    args = parse_args()

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
            

if __name__ == "__main__":
    main()

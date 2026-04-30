# models/trainer.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from models.model_loader import instantiate_model
from utils.metrics_utils import pretty_print_metadata
from utils.file_saver import safe_save_path
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def train_autoencoder(input_path, output_path=None):
    """
    Trains an autoencoder model on a given dataset.

    Parameters:
        input_path (str): Path to the CSV input file.
        output_path (str, optional): Directory to save the trained model files.

    Returns:
        dict: Evaluation metrics computed on validation data.
    """
    logger.info("Starting Autoencoder training on: %s", input_path)

    if not os.path.exists(input_path):
        logger.error("CSV input file not found at %s", input_path)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("Loaded CSV is empty: %s", input_path)
        return

    logger.info("Loaded training data with shape: %s", df.shape)

    if "label" in df.columns:
        X = df.drop(columns=["label"]).values
    else:
        X = df.values

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    model = instantiate_model("autoencoder", input_dim=X.shape[1])
    model.train(X_train, X_val=X_val)

    metrics = model.evaluate(X_val)
    model_dir = output_path or config['training']['save_dir'] + "autoencoder/autoencoder_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nAutoencoder evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    return metrics


def train_random_forest(input_path, output_path=None):
    """
    Trains a Random Forest model on a given dataset.

    Parameters:
        input_path (str): Path to the CSV input file.
        output_path (str, optional): Directory to save the trained model files.

    Returns:
        dict: Evaluation metrics computed on validation data.
    """
    logger.info("Starting Random Forest training on: %s", input_path)

    if not os.path.exists(input_path):
        logger.error("CSV input file not found at %s", input_path)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("Loaded CSV is empty: %s", input_path)
        return

    if "label" not in df.columns:
        logger.error("Missing 'label' column for supervised training.")
        return

    logger.info("Loaded training data with shape: %s", df.shape)

    y = df["label"].values
    X = df.drop(columns=["label"]).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = instantiate_model("random_forest", input_dim=X.shape[1])
    model.train(X_train, y=y_train)

    metrics = model.evaluate(X_val, y_val, log_metrics=False)
    model_dir = output_path or config['training']['save_dir'] + "random_forest/random_forest_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nRandom Forest evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    plot_path = os.path.join(model_dir, "evaluation_report.png")
    model.plot(X_val, y_val, output_path=plot_path)

    return metrics


def train_svm(input_path, output_path=None):
    """
    Trains an SVM model on a given dataset.

    Parameters:
        input_path (str): Path to the CSV input file.
        output_path (str, optional): Directory to save the trained model files.

    Returns:
        dict: Evaluation metrics computed on validation data.
    """
    logger.info("Starting SVM training on: %s", input_path)

    if not os.path.exists(input_path):
        logger.error("CSV input file not found at %s", input_path)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("Loaded CSV is empty: %s", input_path)
        return

    if "label" not in df.columns:
        logger.error("Missing 'label' column for supervised training.")
        return

    logger.info("Loaded training data with shape: %s", df.shape)

    y = df["label"].values
    X = df.drop(columns=["label"]).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = instantiate_model("svm", input_dim=X.shape[1])
    model.train(X_train, y=y_train)

    metrics = model.evaluate(X_val, y_val, log_metrics=False)
    model_dir = output_path or config['training']['save_dir'] + "svm/svm_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nSVM evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    plot_path = os.path.join(model_dir, "evaluation_report.png")
    model.plot(X_val, y_val, output_path=plot_path)

    return metrics


def run_train_model(args):
    """
    Command-line interface handler for model training.

    Parameters:
        args: Parsed command-line arguments containing 'model', 'input', and 'output' options.
    
    This dispatcher handles the flow of the following training operations:
        - Training an Autoencoder model on a given dataset.
        - Training a Random Forest model on a given dataset.
        - Training an SVM model on a given dataset.

    Uses config defaults where needed.
    """
    model_type = args.model or config['training']['model_type']
    input_path = args.input or config['training']['input']
    output_path = args.output or None

    dispatch = {
        "autoencoder": train_autoencoder,
        "random_forest": train_random_forest,
        "svm": train_svm
    }

    if model_type not in dispatch:
        logger.error("Unknown model type: %s", model_type)
        return

    logger.info("Dispatching training for model: %s", model_type)
    dispatch[model_type](input_path, output_path)

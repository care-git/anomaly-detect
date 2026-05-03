# models/trainer.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from models.model_loader import instantiate_model
from utils.metrics_utils import pretty_print_metadata, plot_reconstruction_loss, plot_cv_results
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
    config = get_config()
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
        train_on_anomalous = config.get("training", {}).get("ae_train_on_anomalous", False)
        if not train_on_anomalous:
            n_before = len(df)
            df = df[df["label"] == 0]
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                logger.warning(
                    "%d anomalous rows detected in labelled dataset and dropped before "
                    "autoencoder training. Set ae_train_on_anomalous: true in config to override.",
                    n_dropped,
                )
        X = df.drop(columns=["label"]).values
    else:
        X = df.values

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    model = instantiate_model("autoencoder", input_dim=X.shape[1])
    model.training_dataset = os.path.abspath(input_path)
    model.train(X_train, X_val=X_val)

    metrics = model.evaluate(X_val)
    model_dir = output_path or config['training']['save_dir'] + "autoencoder/autoencoder_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nAutoencoder evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    recon_plot_path = os.path.join(model_dir, "reconstruction_loss.png")
    recon = model.model.predict(model.scaler.transform(X_val), verbose=0)
    mse = np.mean(np.square(model.scaler.transform(X_val) - recon), axis=1)
    plot_reconstruction_loss(mse, threshold=model.threshold, output_path=recon_plot_path)

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
    config = get_config()
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
    model.training_dataset = os.path.abspath(input_path)
    model.feature_names = list(df.drop(columns=["label"]).columns)
    model.train(X_train, y=y_train)

    metrics = model.evaluate(X_val, y_val, log_metrics=False)
    model_dir = output_path or config['training']['save_dir'] + "random_forest/random_forest_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nRandom Forest evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    plot_path = os.path.join(model_dir, "evaluation_report.png")
    feature_names = list(df.drop(columns=["label"]).columns)
    model.plot(X_val, y_val, output_path=plot_path, feature_names=feature_names)

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
    config = get_config()
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
    model.training_dataset = os.path.abspath(input_path)
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


def cross_validate_model(model_type: str, input_path: str, k: int = 5) -> dict:
    """
    Runs k-fold cross-validation for the specified model type and returns
    mean ± std for each evaluation metric across folds.

    For the autoencoder (unsupervised) the fold trains on all features and
    evaluates reconstruction MSE. Labels are used for classification metrics
    only when present. For Random Forest and SVM, StratifiedKFold is used
    with labels required.

    Parameters:
        model_type (str): One of 'autoencoder', 'random_forest', 'svm'.
        input_path (str): Path to the labelled CSV file.
        k (int): Number of folds.

    Returns:
        dict: {'metric_name': {'mean': float, 'std': float}, 'k': k, 'model_type': model_type}
    """
    if not os.path.exists(input_path):
        logger.error("CV input file not found: %s", input_path)
        return {}

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("CV input file is empty: %s", input_path)
        return {}

    has_labels = "label" in df.columns
    supervised = model_type in ("random_forest", "svm")

    if supervised and not has_labels:
        logger.error("Supervised CV requires a 'label' column.")
        return {}

    X = df.drop(columns=["label"]).values if has_labels else df.values
    y = df["label"].values if has_labels else None

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) if (supervised and has_labels) \
               else KFold(n_splits=k, shuffle=True, random_state=42)

    all_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y if supervised else None), start=1):
        logger.info("CV fold %d/%d", fold_idx, k)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx] if y is not None else None
        y_val = y[val_idx] if y is not None else None

        model = instantiate_model(model_type, input_dim=X.shape[1])

        if model_type == "autoencoder":
            X_tr = X_train[y_train == 0] if y_train is not None else X_train
            X_tr, X_inner_val = train_test_split(X_tr, test_size=0.1, random_state=42)
            model.train(X_tr, X_val=X_inner_val)
            fold_metrics = model.evaluate(X_val, y_true=y_val)
        else:
            model.train(X_train, y=y_train)
            fold_metrics = model.evaluate(X_val, y_val)

        # Drop None values
        all_metrics.append({k_: v for k_, v in fold_metrics.items() if v is not None})

    if not all_metrics:
        return {}

    metric_keys = all_metrics[0].keys()
    results: dict = {"model_type": model_type, "k_folds": k}
    for metric in metric_keys:
        values = [m[metric] for m in all_metrics if metric in m]
        results[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    logger.info("Cross-validation complete for %s over %d folds.", model_type, k)
    return results


def run_train_model(args):
    """
    Command-line interface handler for model training.

    Parameters:
        args: Parsed command-line arguments containing 'model', 'input', 'output', 'cv', and 'cv_folds' options.

    This dispatcher handles the flow of the following training operations:
        - Training an Autoencoder model on a given dataset.
        - Training a Random Forest model on a given dataset.
        - Training an SVM model on a given dataset.
        - Running k-fold cross-validation for any of the above when --cv is passed.

    Uses config defaults where needed.
    """
    config = get_config()
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

    if getattr(args, "cv", False):
        k = getattr(args, "cv_folds", 5)
        logger.info("Running %d-fold cross-validation for model: %s", k, model_type)
        cv_results = cross_validate_model(model_type, input_path, k=k)
        pretty_print_metadata(cv_results)

        if output_path:
            plot_cv_results(cv_results, title=f"{model_type} {k}-Fold CV Results",
                            output_path=os.path.join(output_path, "cv_results.png"))
    else:
        logger.info("Dispatching training for model: %s", model_type)
        dispatch[model_type](input_path, output_path)

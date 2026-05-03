# models/benchmark.py

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.model_loader import instantiate_model
from utils.gpu_utils import setup_gpu, release_gpu_memory
from utils.metrics_utils import plot_classification_report
from utils.file_saver import ensure_dir
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__, "INFO")

SUPERVISED_MODELS = ("random_forest", "svm")
ALL_MODELS = ("autoencoder", "random_forest", "svm")
COMPARISON_COLS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


def benchmark_models(input_path: str, output_dir: str = None, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """
    Trains all three models on the same train split and evaluates each on the
    same held-out test split. Requires a labelled CSV as input.

    The autoencoder is trained unsupervised (labels stripped from X) but
    evaluated against ground-truth labels so all models are compared on equal
    footing.

    Parameters:
        input_path (str): Path to a labelled CSV file (must contain a 'label' column).
        output_dir (str, optional): Directory to save comparison table and plots.
        test_size (float): Fraction of data held out for evaluation (default: 0.2).
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Comparison table with one row per model and one column per metric.
    """
    if not os.path.exists(input_path):
        logger.error("Benchmark input file not found: %s", input_path)
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("Benchmark input file is empty.")
        return pd.DataFrame()

    if "label" not in df.columns:
        logger.error("Benchmark requires a 'label' column for evaluation.")
        return pd.DataFrame()

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    feature_names = list(df.drop(columns=["label"]).columns)
    rows = []

    setup_gpu()

    for model_type in ALL_MODELS:
        logger.info("Benchmarking model: %s", model_type)
        try:
            model = instantiate_model(model_type, input_dim=X.shape[1])

            if model_type == "autoencoder":
                X_normal = X_train[y_train == 0]
                X_tr, X_inner_val = train_test_split(X_normal, test_size=0.1, random_state=random_state)
                model.train(X_tr, X_val=X_inner_val)
                metrics = model.evaluate(X_test, y_true=y_test)

                if output_dir:
                    y_pred = model.predict(X_test)
                    plot_classification_report(
                        metrics, y_test, y_pred,
                        title="Autoencoder Evaluation",
                        output_path=os.path.join(output_dir, "autoencoder_evaluation.png")
                    )
            else:
                if model_type == "random_forest":
                    model.feature_names = feature_names
                model.train(X_train, y=y_train)
                metrics = model.evaluate(X_test, y_test)

                if output_dir:
                    y_pred = model.predict(X_test)
                    plot_classification_report(
                        metrics, y_test, y_pred,
                        title=f"{model_type.replace('_', ' ').title()} Evaluation",
                        output_path=os.path.join(output_dir, f"{model_type}_evaluation.png")
                    )

            model.training_dataset = os.path.abspath(input_path)
            row = {"model": model_type}
            for k in COMPARISON_COLS:
                v = metrics.get(k)
                if v is not None:
                    row[k] = round(float(v), 4)
            rows.append(row)

        except Exception as e:
            logger.error("Benchmark failed for %s: %s", model_type, e)
            rows.append({"model": model_type, "error": str(e)})
        finally:
            release_gpu_memory()

    comparison = pd.DataFrame(rows).set_index("model")
    display = comparison.fillna("N/A")

    logger.info("\nBenchmark Results:\n%s", display.to_string())

    if output_dir:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        display.to_csv(csv_path)
        logger.info("Benchmark table saved to: %s", csv_path)

        json_path = os.path.join(output_dir, "benchmark_results.json")
        with open(json_path, "w") as f:
            json.dump(display.reset_index().to_dict(orient="records"), f, indent=2)
        logger.info("Benchmark JSON saved to: %s", json_path)

    return comparison


def run_benchmark(args) -> None:
    """
    Command-line interface handler for the benchmark subcommand.

    Parameters:
        args: Parsed arguments containing 'input' and 'output' options.
    """
    config = get_config()
    input_path = args.input
    output_dir = args.output or os.path.join(config['training']['save_dir'], "benchmark")

    ensure_dir(output_dir)
    benchmark_models(input_path, output_dir=output_dir)

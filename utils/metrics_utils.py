# utils/metrics_utils.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.logger import get_logger

logger = get_logger(__name__, "INFO")


def pretty_print_metadata(metadata: dict, indent: int = 4):
    """
    Makes model metadata and metrics pretty (formatted JSON).

    Parameters:
        metadata (dict): Dictionary containing model metadata and metrics.
        indent (int): Number of spaces for indentation (default is 4).
    """
    if not metadata:
        logger.warning("[!] No metadata found.")
        return
    print(json.dumps(metadata, indent=indent))
    

def _ensure_output_dir(output_path: str) -> None:
    dir_part = os.path.dirname(output_path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)


def plot_classification_report(metrics: dict, y_true, y_pred, title: str = "Model Evaluation", output_path: str = None):
    """
    Visualises model metrics using a bar chart and confusion matrix.

    Parameters:
        metrics (dict): Dictionary with 'accuracy', 'precision', 'recall', and 'f1_score'.
        y_true (list or np.ndarray): Ground truth labels.
        y_pred (list or np.ndarray): Predicted labels.
        title (str, optional): Title for the entire plot.
        output_path (str, optional): Destination file path to save to, instead of displaying plot.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    keys = ["accuracy", "precision", "recall", "f1_score"]
    values = [metrics.get(k, 0) for k in keys]
    sns.barplot(x=keys, y=values, palette="pastel", ax=axes[0])
    axes[0].set_title("Classification Metrics")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=axes[1])
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        _ensure_output_dir(output_path)
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: list, title: str = "Feature Importance", output_path: str = None, top_n: int = 20):
    """
    Plots a horizontal bar chart of Random Forest feature importances.

    Parameters:
        importances (np.ndarray): Array of feature importance scores.
        feature_names (list): Corresponding feature names.
        title (str, optional): Plot title.
        output_path (str, optional): Destination file path to save to, instead of displaying.
        top_n (int): Maximum number of features to display (sorted by importance).
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(4, top_n // 2)))
    sns.barplot(x=top_scores, y=top_names, palette="Blues_d", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    if output_path:
        _ensure_output_dir(output_path)
        plt.savefig(output_path)
        logger.info("Feature importance plot saved to: %s", output_path)
    else:
        plt.show()

    plt.close()


def plot_reconstruction_loss(mse: np.ndarray, y_true=None, threshold: float = None, title: str = "Reconstruction Loss Distribution", output_path: str = None):
    """
    Plots the distribution of per-sample reconstruction error (MSE) from the autoencoder.

    Parameters:
        mse (np.ndarray): Per-sample MSE values.
        y_true (np.ndarray, optional): Ground truth labels for colour-coding normal vs anomalous.
        threshold (float, optional): Anomaly threshold to draw as a vertical line.
        title (str, optional): Plot title.
        output_path (str, optional): Destination file path to save to, instead of displaying.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    if y_true is not None:
        y_true = np.asarray(y_true)
        sns.kdeplot(mse[y_true == 0], label="Normal", fill=True, ax=ax)
        sns.kdeplot(mse[y_true == 1], label="Anomalous", fill=True, ax=ax)
        ax.legend()
    else:
        sns.kdeplot(mse, fill=True, ax=ax)

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})")
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("MSE (Reconstruction Loss)")
    ax.set_ylabel("Density")
    plt.tight_layout()

    if output_path:
        _ensure_output_dir(output_path)
        plt.savefig(output_path)
        logger.info("Reconstruction loss plot saved to: %s", output_path)
    else:
        plt.show()

    plt.close()


def plot_cv_results(cv_results: dict, title: str = "Cross-Validation Results", output_path: str = None):
    """
    Plots mean ± std of cross-validation metrics as a bar chart with error bars.

    Parameters:
        cv_results (dict): Dict of metric_name -> {'mean': float, 'std': float}.
        title (str, optional): Plot title.
        output_path (str, optional): Destination file path to save to, instead of displaying.
    """
    metric_keys = [k for k in cv_results if isinstance(cv_results[k], dict) and "mean" in cv_results[k]]
    means = [cv_results[k]["mean"] for k in metric_keys]
    stds = [cv_results[k]["std"] for k in metric_keys]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(metric_keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette("pastel", len(metric_keys)))
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        _ensure_output_dir(output_path)
        plt.savefig(output_path)
        logger.info("CV results plot saved to: %s", output_path)
    else:
        plt.show()

    plt.close()
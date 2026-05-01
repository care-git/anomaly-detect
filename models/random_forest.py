# models/random_forest.py

import os
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

from models.base_model import BaseModel
from utils.metrics_utils import plot_classification_report, plot_feature_importance
from utils.file_saver import save_pickle, save_json, ensure_dir
from utils.progress import TrainingSpinner
from utils.config_loader import get_config
from utils.logger import get_logger

try:
    from cuml.ensemble import RandomForestClassifier as _cuRF
    _CUML_AVAILABLE = True
except ImportError:
    _CUML_AVAILABLE = False

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))

class RandomForestModel(BaseModel):
    """
    Random Forest classifier implementation of BaseModel.

    Suitable for supervised learning on labelled datasets.
    """

    def __init__(self, **kwargs):
        """
        Initialises the Random Forest model with filtered keyword arguments.

        Parameters:
            **kwargs: Hyperparameters for RandomForestClassifier. Unrelated keys are ignored.
        """
        # Strip out unrelated kwargs like 'input_dim'
        rf_kwargs = {k: v for k, v in kwargs.items() if k in RandomForestClassifier().get_params()}
        self.model = RandomForestClassifier(**rf_kwargs)
        self.input_dim = None
        self.metadata = {}
        self._using_cuml = False
        logger.info("Initialised Random Forest model with params: %s", rf_kwargs)

    def train(self, X, y=None, **kwargs):
        """
        Trains the Random Forest model on labelled data.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.
            y (np.ndarray or pd.Series): Class labels.
            **kwargs: Additional hyperparameters for RandomForestClassifier.
        """
        if y is None:
            raise ValueError("Supervised training requires labels (y).")

        self.input_dim = X.shape[1]
        training_cfg = get_config().get("training", {})
        n_estimators = training_cfg.get("n_estimators", 100)
        use_gpu = training_cfg.get("use_gpu", False)

        if use_gpu and _CUML_AVAILABLE:
            logger.info("Training Random Forest [cuML GPU] with %d estimators on %d samples", n_estimators, len(X))
            self.model = _cuRF(n_estimators=n_estimators)
            with TrainingSpinner("Training Random Forest [GPU]") as spinner:
                self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
            self._using_cuml = True
        else:
            if use_gpu and not _CUML_AVAILABLE:
                logger.warning("use_gpu=True but cuML not found — falling back to sklearn RandomForest.")
            logger.info("Training %d trees [sklearn CPU] on %d samples", n_estimators, len(X))
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                warm_start=True,
                **{k: v for k, v in kwargs.items() if k in RandomForestClassifier().get_params()}
            )
            # warm_start=True lets us increment n_estimators by 1 each call so the
            # spinner can show per-tree progress; each fit() adds only one new tree.
            with TrainingSpinner("Training Random Forest") as spinner:
                for i in range(1, n_estimators + 1):
                    self.model.n_estimators = i
                    self.model.fit(X, y)
                    spinner.update({"tree": f"{i}/{n_estimators}"})
            self._using_cuml = False

        logger.info("Training complete.")

    def predict(self, X):
        """
        Generates class predictions from the trained model.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        logger.info("Predicting using trained Random Forest model on %d samples", len(X))
        X_in = np.asarray(X, dtype=np.float32) if self._using_cuml else X
        return np.asarray(self.model.predict(X_in))

    def evaluate(self, X, y_true, log_metrics=False):
        """
        Evaluates the classifier on labelled data.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.
            y_true (np.ndarray or pd.Series): Ground truth labels.
            log_metrics (bool): Whether to log classification report.

        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1 score.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        X_in = np.asarray(X, dtype=np.float32) if self._using_cuml else X
        y_pred = np.asarray(self.model.predict(X_in))
        y_proba = np.asarray(self.model.predict_proba(X_in)[:, 1])
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        }

        logger.info("Random Forest model evaluation complete.")

        if log_metrics:
            logger.info("Random Forest model evaluation:\n%s", classification_report(y_true, y_pred, zero_division=0))

        return metrics
    
    def plot(self, X_val, y_val, output_path=None, title=None, feature_names=None):
        """
        Generates and saves a classification report plot and feature importance chart.

        Parameters:
            X_val (np.ndarray): Validation feature set.
            y_val (np.ndarray): Validation labels.
            output_path (str, optional): File path to save the classification report plot.
            title (str, optional): Title for the classification report plot.
            feature_names (list, optional): Feature names for the importance chart.
        """
        y_pred = self.model.predict(X_val)
        metrics = self.evaluate(X_val, y_val, log_metrics=True)
        title = title or f"{self.metadata.get('model_type', 'Model')} Evaluation"

        plot_classification_report(metrics, y_val, y_pred, title=title, output_path=output_path)

        importances = np.asarray(self.model.feature_importances_)
        names = feature_names or [f"feature_{i}" for i in range(len(importances))]
        importance_path = output_path.replace(".png", "_feature_importance.png") if output_path else None
        plot_feature_importance(importances, names, title=f"{title} — Feature Importance", output_path=importance_path)

    def save(self, path,  metrics=None):
        """
        Saves the model, metadata, and optionally evaluation metrics.

        Parameters:
            path (str): Directory to save model files.
            metrics (dict, optional): Evaluation metrics to store in metadata.
        """
        ensure_dir(path)

        model_path = os.path.join(path, "model.pkl")
        save_pickle(self.model, model_path)

        self.metadata = {
            "model_type": "random_forest",
            "model_path": model_path,
            "input_dim": self.input_dim,
            "using_cuml": self._using_cuml,
            "evaluation_metrics": metrics or {}
        }

        metadata_path = os.path.join(path, 'metadata.json')
        save_json(self.metadata, metadata_path)

        logger.info("Random Forest model, metadata, and metrics plot saved to: %s", path)

    def load(self, path):
        """
        Loads the model and metadata from disk.

        Parameters:
            path (str): Directory to load model files from.
        """
        model_path = os.path.join(path, 'model.pkl')
        metadata_path = os.path.join(path, 'metadata.json')

        self.model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.input_dim = self.metadata.get("input_dim")
            self._using_cuml = self.metadata.get("using_cuml", False)
            if self._using_cuml and not _CUML_AVAILABLE:
                logger.warning("Model was trained with cuML but cuML is not installed — predictions may fail.")

        logger.info("Random Forest model loaded from: %s", path)

    def get_metadata(self, path) -> dict:
        """
        Returns stored model metadata.

        Parameters:
            path (str): Base path to construct fallback metadata if none exists.

        Returns:
            dict: Metadata dictionary.
        """
        return self.metadata or {
            "model_type": "random_forest",
            "model_path": os.path.join(path, "model.pkl"),
            "input_dim": self.input_dim,
            "evaluation_metrics": {}
        }

# models/svm.py

import os
import joblib
import json
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from models.base_model import BaseModel
from utils.metrics_utils import plot_classification_report
from utils.progress import single_bar
from utils.file_saver import save_pickle, save_json, ensure_dir
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))

class SVMModel(BaseModel):
    """
    Support Vector Machine classifier implementation of BaseModel.

    Suitable for supervised learning on labelled datasets.
    """

    def __init__(self, **kwargs):
        """
        Initialises the SVM model with configurable parameters.

        Parameters:
            **kwargs: Parameters passed to sklearn's SVC.
        """
        # Strip unrelated kwargs like rf
        svm_kwargs = {k: v for k, v in kwargs.items() if k in SVC().get_params()}
        self.model = SVC(probability=True, **svm_kwargs)
        self.input_dim = None
        self.metadata = {}
        logger.info("Initialised SVM model with params: %s", svm_kwargs)

    def train(self, X, y=None, **kwargs):
        """
        Trains the SVM model on the labelled data.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.
            y (np.ndarray or pd.Series): Class labels.
            **kwargs: Ignored (included for interface consistency).
        """
        if y is None:
            raise ValueError("Supervised training requires labels (y).")
        
        self.input_dim = X.shape[1]
        logger.info("Training SVM model on data with %d samples", len(X))

        with single_bar("Training SVM", unit="step") as update:
            self.model.fit(X, y)
            update()

    def predict(self, X):
        """
        Generates class predictions from the trained model.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        logger.info("Predicting using SVM model on %d samples", len(X))
        return self.model.predict(X)

    def evaluate(self, X, y_true, log_metrics=False):
        """
        Evaluates the classifier on labelled data.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.
            y_true (np.ndarray or pd.Series): Ground truth labels.
            log_metrics (bool): Whether to log the classification report.

        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1 score.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        y_pred = self.model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        logger.info("SVM model evaluation complete.")

        if log_metrics:
            logger.info("SVM model evaluation:\n%s", classification_report(y_true, y_pred, zero_division=0))

        return metrics
    
    def plot(self, X_val, y_val, output_path=None, title=None):
        """
        Generates and saves a classification report plot.

        Parameters:
            X_val (np.ndarray): Validation feature set.
            y_val (np.ndarray): Validation labels.
            output_path (str, optional): File path to save the plot.
            title (str, optional): Title for the plot.
        """
        y_pred = self.model.predict(X_val)
        metrics = self.evaluate(X_val, y_val, log_metrics=True)
        title = title or f"{self.metadata.get('model_type', 'Model')} Evaluation"

        plot_classification_report(metrics, y_val, y_pred, title=title, output_path=output_path)

    def save(self, path, metrics=None):
        """
        Saves the model, metadata, and optionally evaluation metrics.

        Parameters:
            path (str): Directory to save model files.
            metrics (dict, optional): Evaluation results to store in metadata.
        """
        ensure_dir(path)

        model_path = os.path.join(path, "model.pkl")
        save_pickle(self.model, model_path)

        self.metadata = {
            "model_type": "svm",
            "model_path": model_path,
            "input_dim": self.input_dim,
            "evaluation_metrics": metrics or {}
        }

        metadata_path = os.path.join(path, 'metadata.json')
        save_json(self.metadata, metadata_path)

        logger.info("SVM model, metadata, and metrics plot saved to: %s", path)

    def load(self, path):
        """
        Loads the model and metadata from disk.

        Parameters:
            path (str): Directory to load model files from.
        """
        model_path = os.path.join(path, "model.pkl")
        metadata_path = os.path.join(path, "metadata.json")

        self.model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.input_dim = self.metadata.get("input_dim")

        logger.info("SVM model loaded from: %s", path)

    def get_metadata(self, path) -> dict:
        """
        Returns stored model metadata.

        Parameters:
            path (str): Base path to construct fallback metadata if none exists.

        Returns:
            dict: Metadata dictionary.
        """
        return self.metadata or {
            "model_type": "svm",
            "model_path": os.path.join(path, "model.pkl"),
            "input_dim": self.input_dim,
            "evaluation_metrics": {}
        }
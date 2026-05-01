# models/svm.py

import os
import joblib
import json
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

try:
    from cuml.svm import SVC as _cuSVC
    _CUML_AVAILABLE = True
except ImportError:
    _CUML_AVAILABLE = False

from models.base_model import BaseModel
from utils.metrics_utils import plot_classification_report
from utils.file_saver import save_pickle, save_json, ensure_dir
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__, "INFO")


class SVMModel(BaseModel):
    """
    Support Vector Machine classifier implementation of BaseModel.

    Input features are always standardised via StandardScaler before fitting
    and inference — this is required for convergence with LinearSVC and
    significantly improves performance with RBF/poly kernels.

    Supports two backends selected via config `training.svm_kernel`:
        - 'linear': uses LinearSVC wrapped in CalibratedClassifierCV for
          probability estimates. Scales to large datasets (O(n) training time).
        - any other value (e.g. 'rbf', 'poly'): uses SVC with that kernel.
          More expressive but O(n²) training time — avoid on large datasets.
    """

    def __init__(self, **kwargs):
        """
        Initialises the SVM model shell. The backend (SVC or LinearSVC) is
        selected during train() based on the config value of svm_kernel.

        Parameters:
            **kwargs: Passed to SVC if the rbf/poly backend is used. Unrelated
                      keys such as 'input_dim' are silently dropped.
        """
        self._init_kwargs = {k: v for k, v in kwargs.items() if k in SVC().get_params()}
        self.model = None
        self.scaler = StandardScaler()
        self.input_dim = None
        self.metadata = {}
        self._use_linear = False
        self._using_cuml = False

    def _build_model(self, use_linear: bool, max_iter: int, kernel: str = "rbf", use_gpu: bool = False):
        """Constructs the appropriate estimator (cuML or sklearn)."""
        if use_gpu and _CUML_AVAILABLE:
            # cuML SVC has native probability support — no CalibratedClassifierCV needed
            return _cuSVC(probability=True, kernel="linear" if use_linear else kernel)
        if use_linear:
            base = LinearSVC(max_iter=max_iter)
            return CalibratedClassifierCV(base, cv=3)
        return SVC(probability=True, **self._init_kwargs)

    def train(self, X, y=None, **kwargs):
        """
        Fits a StandardScaler on the training data then trains the SVM.

        The backend is chosen from config `training.svm_kernel`. Setting it to
        'linear' instantiates a LinearSVC (via CalibratedClassifierCV for
        probability estimates); any other value uses SVC with that kernel.
        Max solver iterations are controlled by config `training.svm_max_iter`.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features (unscaled).
            y (np.ndarray or pd.Series): Class labels.
            **kwargs: Ignored (included for interface consistency).
        """
        if y is None:
            raise ValueError("Supervised training requires labels (y).")

        config = get_config()
        training_cfg = config.get("training", {})
        svm_kernel = training_cfg.get("svm_kernel", "rbf")
        max_iter = training_cfg.get("svm_max_iter", 2000)
        use_gpu = training_cfg.get("use_gpu", False)

        self._use_linear = svm_kernel == "linear"
        self._using_cuml = use_gpu and _CUML_AVAILABLE
        self.input_dim = X.shape[1]
        self.scaler = StandardScaler()
        self.model = self._build_model(self._use_linear, max_iter, kernel=svm_kernel, use_gpu=use_gpu)

        if use_gpu and not _CUML_AVAILABLE:
            logger.warning("use_gpu=True but cuML not found — falling back to sklearn SVM.")

        X_scaled = self.scaler.fit_transform(X)
        if self._using_cuml:
            X_scaled = X_scaled.astype(np.float32)
            y = np.asarray(y, dtype=np.float32)

        if self._using_cuml:
            backend_name = f"cuML SVC (kernel={svm_kernel})"
        elif self._use_linear:
            backend_name = "LinearSVC (CalibratedClassifierCV)"
        else:
            backend_name = f"SVC (kernel={svm_kernel})"

        logger.info("Training SVM [%s] on %d samples", backend_name, len(X))
        self.model.fit(X_scaled, y)
        logger.info("SVM training complete.")

    def predict(self, X):
        """
        Scales input then generates class predictions from the trained model.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features (unscaled).

        Returns:
            np.ndarray: Predicted labels.
        """
        logger.info("Predicting using SVM model on %d samples", len(X))
        X_scaled = self.scaler.transform(X)
        if self._using_cuml:
            X_scaled = X_scaled.astype(np.float32)
        return np.asarray(self.model.predict(X_scaled))

    def evaluate(self, X, y_true, log_metrics=False):
        """
        Scales input then evaluates the classifier on labelled data.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features (unscaled).
            y_true (np.ndarray or pd.Series): Ground truth labels.
            log_metrics (bool): Whether to log the full classification report.

        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, F1, and ROC-AUC.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        X_scaled = self.scaler.transform(X)
        if self._using_cuml:
            X_scaled = X_scaled.astype(np.float32)
        y_pred = np.asarray(self.model.predict(X_scaled))

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        # ROC-AUC via probability estimates when available, decision function otherwise.
        if hasattr(self.model, "predict_proba"):
            y_score = np.asarray(self.model.predict_proba(X_scaled)[:, 1])
        elif hasattr(self.model, "decision_function"):
            y_score = np.asarray(self.model.decision_function(X_scaled))
        else:
            y_score = y_pred
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None

        logger.info("SVM model evaluation complete.")

        if log_metrics:
            logger.info("SVM model evaluation:\n%s", classification_report(y_true, y_pred, zero_division=0))

        return metrics

    def plot(self, X_val, y_val, output_path=None, title=None):
        """
        Generates and saves a classification report plot.

        Parameters:
            X_val (np.ndarray): Validation feature set (unscaled).
            y_val (np.ndarray): Validation labels.
            output_path (str, optional): File path to save the plot.
            title (str, optional): Title for the plot.
        """
        y_pred = self.predict(X_val)
        metrics = self.evaluate(X_val, y_val, log_metrics=True)
        title = title or f"{self.metadata.get('model_type', 'Model')} Evaluation"

        plot_classification_report(metrics, y_val, y_pred, title=title, output_path=output_path)

    def save(self, path, metrics=None):
        """
        Saves the model, scaler, metadata, and optionally evaluation metrics.

        Parameters:
            path (str): Directory to save model files.
            metrics (dict, optional): Evaluation results to store in metadata.
        """
        ensure_dir(path)

        model_path = os.path.join(path, "model.pkl")
        save_pickle(self.model, model_path)

        scaler_path = os.path.join(path, "scaler.pkl")
        save_pickle(self.scaler, scaler_path)

        self.metadata = {
            "model_type": "svm",
            "model_path": model_path,
            "scaler_path": scaler_path,
            "input_dim": self.input_dim,
            "use_linear": self._use_linear,
            "using_cuml": self._using_cuml,
            "evaluation_metrics": metrics or {}
        }

        metadata_path = os.path.join(path, 'metadata.json')
        save_json(self.metadata, metadata_path)

        logger.info("SVM model and scaler saved to: %s", path)

    def load(self, path):
        """
        Loads the model, scaler, and metadata from disk.

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
            self._use_linear = self.metadata.get("use_linear", False)
            self._using_cuml = self.metadata.get("using_cuml", False)
            if self._using_cuml and not _CUML_AVAILABLE:
                logger.warning("Model was trained with cuML but cuML is not installed — predictions may fail.")
            scaler_path = self.metadata.get("scaler_path")
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                logger.warning("Scaler not found in metadata — predictions may be inaccurate.")

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
            "scaler_path": os.path.join(path, "scaler.pkl"),
            "input_dim": self.input_dim,
            "evaluation_metrics": {}
        }

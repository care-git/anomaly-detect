# models/autoencoder.py

import os
import numpy as np
import joblib
import json
from datetime import datetime
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from models.base_model import BaseModel
from utils.file_saver import save_keras_model, save_pickle, save_json, ensure_dir
from utils.gpu_utils import setup_gpu
from utils.metrics_utils import plot_classification_report
from utils.progress import TrainingSpinner
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


class _SpinnerEpochCallback(Callback):
    """Pushes per-epoch metrics to a TrainingSpinner during model.fit()."""

    def __init__(self, spinner: TrainingSpinner, total_epochs: int):
        super().__init__()
        self._spinner = spinner
        self._total = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        stats = {"epoch": f"{epoch + 1}/{self._total}"}
        stats.update({k: f"{v:.4f}" for k, v in (logs or {}).items()})
        self._spinner.update(stats)


class AutoencoderModel(BaseModel):
    """
    Unsupervised anomaly detector using a simple feedforward autoencoder.

    Learns to reconstruct input data and flags anomalies based on reconstruction error.
    """

    def __init__(self, input_dim=None, threshold=None):
        """
        Initialise the autoencoder model and related components.

        Parameters:
            input_dim (int, optional): Dimensionality of input features.
            threshold (float, optional): Anomaly score threshold.
        """
        super().__init__()
        self.model = None
        self.threshold = threshold
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        self.training_dataset = None
        self._trained_at = None
        self._n_training_samples = None
        self._threshold_percentile = 95

    def build_model(self):
        """
        Constructs and compiles the autoencoder architecture.
        
        Returns:
            Model: A compiled Keras autoencoder model.
        """
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        output_layer = Dense(self.input_dim, activation='linear')(decoded)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        return model

    def train(self, X, y=None, X_val=None):
        """
        Trains the autoencoder model using unsupervised learning.
        
        Parameters:
            X (np.array or pd.DataFrame): Training feature data.
            y (ignored): Included for compatibility; not used.
            X_val (np.ndarray or pd.DataFrame, optional): Optional validation data.

        Raises:
            ValueError: If input data contains NaNs or Infs after scaling.
        """
        setup_gpu()
        self.input_dim = X.shape[1]
        self.model = self.build_model()

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            raise ValueError("Training input contains NaN or Inf after scaling.")

        logger.info("Training autoencoder | %d samples | %d features", X.shape[0], X.shape[1])

        cfg = get_config().get("training", {})
        patience = cfg.get("ae_patience", 5)
        max_epochs = cfg.get("ae_epochs", 100)
        batch_size = cfg.get("ae_batch_size", 32)
        with TrainingSpinner("Training Autoencoder") as spinner:
            self.model.fit(
                X_scaled, X_scaled,
                epochs=max_epochs,
                batch_size=batch_size,
                validation_data=(X_val_scaled, X_val_scaled) if X_val_scaled is not None else None,
                callbacks=[
                    EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True),
                    _SpinnerEpochCallback(spinner, total_epochs=max_epochs),
                ],
                verbose=0
            )

        # Set anomaly threshold based on reconstruction error distribution
        recon = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - recon), axis=1)
        
        percentile = cfg.get("ae_threshold_percentile", 95)
        self.threshold = float(np.percentile(mse, percentile))
        self._threshold_percentile = percentile
        self._n_training_samples = X.shape[0]
        self._trained_at = datetime.now().isoformat()

        logger.info(
            "Autoencoder trained - threshold: %.4f (p%d) | train MSE mean: %.4f | std: %.4f",
            self.threshold, percentile, mse.mean(), mse.std(),
        )


    def predict(self, X):
        """
        Predicts binary anomaly labels based on reconstruction error.

        Stores per-sample MSE in self.last_mse so callers can retrieve anomaly
        scores without running inference a second time.

        Parameters:
            X (np.ndarray or pd.DataFrame): Feature data to score.

        Returns:
            np.ndarray: Binary predictions (0 = normal, 1 = anomalous).
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {X.shape[1]}")

        X_scaled = self.scaler.transform(X)
        logger.info("Predicting on %d samples", X.shape[0])
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.last_mse = mse
        return (mse > self.threshold).astype(int)


    def evaluate(self, X, y_true=None):
        """
        Evaluates the model using reconstruction error and optional labels.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input data.
            y_true (np.ndarray or list, optional): Ground truth labels.

        Returns:
            dict: Evaluation metrics and statistics.
        """
        if self.model is None or self.scaler is None:
            logger.error("Model or Model scaler not loaded.")

        X_scaled = self.scaler.transform(X)

        recon = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - recon), axis=1)
        mae = np.mean(np.abs(X_scaled - recon), axis=1)

        results = {
            "mse_mean": float(mse.mean()),
            "mse_std": float(mse.std()),
            "mse_median": float(np.median(mse)),
            "mse_iqr": float(np.percentile(mse, 75) - np.percentile(mse, 25)),
            "mse_p95": float(np.percentile(mse, 95)),
            "mae_mean": float(mae.mean()),
            "mae_std": float(mae.std()),
            "mae_median": float(np.median(mae)),
            "mae_iqr": float(np.percentile(mae, 75) - np.percentile(mae, 25)),
        }

        logger.info(
            "Eval MSE - mean: %.4f | median: %.4f | std: %.4f | IQR: %.4f | p95: %.4f",
            results["mse_mean"], results["mse_median"], results["mse_std"],
            results["mse_iqr"], results["mse_p95"],
        )
        logger.info(
            "Eval MAE - mean: %.4f | median: %.4f | std: %.4f | IQR: %.4f",
            results["mae_mean"], results["mae_median"], results["mae_std"], results["mae_iqr"],
        )

        if y_true is not None:
            y_true = np.asarray(y_true)
            y_pred = (mse > self.threshold).astype(int)

            avg_mse_normal = float(mse[y_true == 0].mean()) if (y_true == 0).any() else None
            avg_mse_anomalous = float(mse[y_true == 1].mean()) if (y_true == 1).any() else None
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = float(roc_auc_score(y_true, mse)) if len(np.unique(y_true)) > 1 else None

            if avg_mse_normal is not None and avg_mse_anomalous is not None:
                logger.info(
                    "Reconstruction - normal MSE: %.4f | anomalous MSE: %.4f",
                    avg_mse_normal, avg_mse_anomalous,
                )
            logger.info(
                "Evaluation - accuracy: %.4f | precision: %.4f | recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
                accuracy, precision, recall, f1, roc_auc if roc_auc is not None else 0.0,
            )

            results.update({
                "avg_mse_normal": avg_mse_normal,
                "avg_mse_anomalous": avg_mse_anomalous,
                "avg_mae_normal": float(mae[y_true == 0].mean()) if (y_true == 0).any() else None,
                "avg_mae_anomalous": float(mae[y_true == 1].mean()) if (y_true == 1).any() else None,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
            })

        return results


    def plot(self, X, y_true, output_path=None):
        """
        Saves an evaluation plot (classification metrics bar chart + confusion matrix).

        Parameters:
            X (np.ndarray or pd.DataFrame): Input data.
            y_true (np.ndarray): Ground truth labels.
            output_path (str, optional): File path to save the plot.
        """
        y_pred = self.predict(X)
        metrics = self.evaluate(X, y_true=y_true)
        plot_classification_report(
            metrics, y_true, y_pred,
            title="Autoencoder Evaluation",
            output_path=output_path,
        )


    def save(self, path, metrics=None):
        """
        Saves the trained autoencoder model and metadata to disk.

        Parameters:
            path (str): Directory path to save model components.
            metrics (dict, optional): Evaluation metrics to include in metadata.    
        """
        ensure_dir(path)

        model_path = os.path.join(path, 'model.keras')
        save_keras_model(self.model, model_path)
        
        scaler_path = os.path.join(path, 'scaler.pkl')
        save_pickle(self.scaler, scaler_path)

        architecture = [self.input_dim] + [
            layer.units for layer in self.model.layers if hasattr(layer, "units")
        ]

        self.metadata = {
            "model_type": "autoencoder",
            "model_path": model_path,
            "scaler_path": scaler_path,
            "input_dim": self.input_dim,
            "architecture": architecture,
            "threshold": self.threshold,
            "threshold_percentile": self._threshold_percentile,
            "trained_at": self._trained_at,
            "n_training_samples": self._n_training_samples,
            "training_dataset": self.training_dataset,
            "evaluation_metrics": metrics or {},
        }

        metadata_path = os.path.join(path, 'metadata.json')
        save_json(self.metadata, metadata_path)

        logger.info("Saved to: %s", path)


    def load(self, path):
        """
        Loads the trained model, scaler, and metadata from disk.

        Parameters:
            path (str): Directory where model components are stored.
        """
        self.model = load_model(os.path.join(path, 'model.keras'))

        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        self.metadata_path = metadata_path

        # Restore attributes
        self.input_dim = self.metadata.get("input_dim")
        self.threshold = self.metadata.get("threshold")
        self.scaler = joblib.load(self.metadata.get("scaler_path"))

        logger.info("Loaded from: %s", path)


    def get_metadata(self, path) -> dict:
        """
        Returns metadata dictionary for the trained model.

        Parameters:
            path (str, optional): Override path (not required).

        Returns:
            dict: Metadata including paths, threshold, and metrics.
        """
        return self.metadata or {
            "model_type": "autoencoder",
            "model_path": os.path.join(path, "model.keras"),
            "input_dim": self.input_dim,
            "evaluation_metrics": {}
        }

# models/autoencoder.py

import os
import numpy as np
import joblib
import json
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.base_model import BaseModel
from utils.file_saver import save_keras_model, save_pickle, save_json, ensure_dir
from utils.progress import single_bar
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


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
        self.input_dim = X.shape[1]
        self.model = self.build_model()
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            raise ValueError("Training input contains NaN or Inf after scaling.")

        logger.info("Training Autoencoder on %d samples with %d features", X.shape[0], X.shape[1])

        patience = 5
        max_epochs = 100
        with single_bar("Training Autoencoder", unit="stage") as update:
            self.model.fit(
                X_scaled, X_scaled,
                epochs=max_epochs,
                batch_size=32,
                validation_data=(X_val_scaled, X_val_scaled) if X_val_scaled is not None else None,
                callbacks=[
                    EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True),
                    ],
                verbose=0
            )
            update()

        # Set anomaly threshold based on reconstruction error distribution
        recon = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - recon), axis=1)
        
        logger.info("[Train MSE] mean: %.6f | std: %.6f", mse.mean(), mse.std())
        self.threshold = mse.mean() + 3 * mse.std()
        
        logger.info("Training complete. Anomaly threshold set to %.6f", self.threshold)


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
        logger.info("Predicting anomalies on %d samples", X.shape[0])
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

        logger.info("[Eval MSE] mean: %.6f | std: %.6f", mse.mean(), mse.std())

        results = {
            "mse_mean": mse.mean(),
            "mse_std": mse.std(),
        }

        if y_true is not None:
            y_pred = (mse > self.threshold).astype(int)
            
            results.update({
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "avg_mse_normal": float(mse[y_true == 0].mean()) if (y_true == 0).any() else None,
                "avg_mse_anomalous": float(mse[y_true == 1].mean()) if (y_true == 1).any() else None,
            })

        return results


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

        self.metadata = {
            "model_type": "autoencoder",
            "model_path": model_path,
            "scaler_path": scaler_path,
            "threshold": self.threshold,
            "input_dim": self.input_dim,
            "evaluation_metrics": metrics or {}
        }

        metadata_path = os.path.join(path, 'metadata.json')
        save_json(self.metadata, metadata_path)

        logger.info("Autoencoder model and metadata saved to: %s", path)


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

        logger.info("Autoencoder model loaded from: %s", path)


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

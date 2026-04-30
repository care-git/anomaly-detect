# models/base_model.py

from abc import ABC, abstractmethod
from typing import Any, Union, Optional
import numpy as np
import pandas as pd
import logging
import joblib


class BaseModel(ABC):
    """
    Abstract base class for all supported ML models in the pipeline.

    All custom models must inherit from this class and implement the required methods.
    The class provides default utility functions for logging, joblib-based persistence, and optional metadata.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Initialises the base model with optional configuration and logger.

        Parameters:
            config (Optional[dict]): Optional dictionary of configuration parameters (e.g., thresholds or paths).
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None, X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> None:
        """
        Trains the model using the given dataset.

        Parameters:
            X (Union[np.ndarray, pd.Dataframe]): Input features.
            y (Any, optional): Labels for supervised models.
            X_val (Optional[Union[np.ndarray, pd.DataFrame]]): Optional validation data.
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generates predictions or anomaly scores from the model.

        Parameters:
            X (Union[np.ndarray, pd.DataFrame]): Input features.

        Returns:
            np.ndarray: Array of predictions or anomaly scores.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to a specified file path.

        Parameters:
            path (str): Destination file path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the model from the specified file path.

        Parameters:
            path (str): Path to the saved model.
        """
        pass

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None) -> np.ndarray:
        """
        Trains the model and immediately predicts on the input data.

        Parameters:
            X Union[np.ndarray, pd.DataFrame]: Input features.
            y (Any | Optional): Labels for supervised models.

        Returns:
            np.ndarray: Array of predictions or anomaly scores.
        """
        self.train(X, y)
        return self.predict(X)

    def get_metadata(self) -> dict:
        """
        Returns optional metadata for the model.

        Can include threshold values, input data dimensions, etc.
        Override in subclasses as needed.

        Returns:
            dict: Dictionary containing metadata.
        """
        return {}

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Any) -> dict:
        """
        Optional method for to evaluate model performance. 
        
        Subclasses can override this to return metrics (e.g., accuracy or F1 scores).

        Parameters:
            X (Union[np.ndarray, pd.DataFrame]): Input features.
            y (Any): Ground truth labels.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        raise NotImplementedError("Evaluation not implemented for this model.")

    def save_with_joblib(self, obj: Any, path: str) -> None:
        """
        Saves any serialisable object using joblib.

        Parameters:
            obj (Any): Object to be saved.
            path (str): Destination file path.
        """
        joblib.dump(obj, path)
        self.logger.info(f"Model saved to {path}")

    def load_with_joblib(self, path: str) -> Any:
        """
        Loads a joblib-saved object from the disk.

        Parameters:
            path (str): Path to the saved object.

        Returns:
            Any: Loaded object.
        """
        self.logger.info(f"Loading model from {path}")
        return joblib.load(path)

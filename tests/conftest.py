# tests/conftest.py
# Shared fixtures available to all test modules.

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def labelled_df():
    """20-row binary-labelled DataFrame — 10 normal (label=0), 10 anomalous (label=1).

    Normal samples cluster near the origin; anomalous samples are offset by 3
    units in every feature dimension, giving clear class separation so supervised
    models can learn a meaningful boundary on a tiny fixture.
    """
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=0.3, size=(10, 3))
    anomalous = rng.normal(loc=3.0, scale=0.3, size=(10, 3))
    X = np.vstack([normal, anomalous])
    y = np.array([0] * 10 + [1] * 10)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["label"] = y
    return df


@pytest.fixture
def normal_df():
    """20-row unlabelled DataFrame containing only normal-traffic-like samples."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0.0, scale=0.3, size=(20, 3))
    return pd.DataFrame(X, columns=["f1", "f2", "f3"])


@pytest.fixture
def labelled_csv(tmp_path, labelled_df):
    """Writes labelled_df to a temporary CSV and returns the path as a string."""
    path = tmp_path / "labelled.csv"
    labelled_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def normal_csv(tmp_path, normal_df):
    """Writes normal_df to a temporary CSV and returns the path as a string."""
    path = tmp_path / "normal.csv"
    normal_df.to_csv(path, index=False)
    return str(path)

# tests/test_models/test_autoencoder.py

import pytest
import numpy as np
from models.autoencoder import AutoencoderModel


def _cfg(percentile=95):
    return {"training": {
        "ae_epochs": 2,
        "ae_patience": 1,
        "ae_batch_size": 32,
        "ae_threshold_percentile": percentile,
    }}


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def test_build_model_input_shape():
    model = AutoencoderModel(input_dim=3)
    net = model.build_model()
    assert net.input_shape[1] == 3


def test_build_model_output_shape_matches_input():
    model = AutoencoderModel(input_dim=5)
    net = model.build_model()
    assert net.output_shape[1] == 5


# ---------------------------------------------------------------------------
# Train + evaluate (unlabelled)
# ---------------------------------------------------------------------------

def test_train_and_evaluate_outputs_reconstruction_metrics(normal_df, monkeypatch):
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg())
    model = AutoencoderModel(input_dim=3)
    model.train(normal_df.values)
    metrics = model.evaluate(normal_df.values)

    for key in ("mse_mean", "mse_std", "mse_median", "mae_mean"):
        assert key in metrics
    assert metrics["mse_mean"] > 0


def test_evaluate_without_labels_excludes_classification_metrics(normal_df, monkeypatch):
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg())
    model = AutoencoderModel(input_dim=3)
    model.train(normal_df.values)
    metrics = model.evaluate(normal_df.values)

    assert "accuracy" not in metrics
    assert "f1_score" not in metrics


# ---------------------------------------------------------------------------
# Train + evaluate (labelled)
# ---------------------------------------------------------------------------

def test_evaluate_with_labels_includes_classification_metrics(labelled_df, monkeypatch):
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg())
    X = labelled_df.drop(columns=["label"]).values
    y = labelled_df["label"].values

    model = AutoencoderModel(input_dim=3)
    model.train(X[y == 0])
    metrics = model.evaluate(X, y_true=y)

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert key in metrics


# ---------------------------------------------------------------------------
# Threshold percentile behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("percentile", [50, 75, 95])
def test_threshold_percentile_stored_in_attribute(percentile, normal_df, monkeypatch):
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg(percentile))
    model = AutoencoderModel(input_dim=3)
    model.train(normal_df.values)
    assert model._threshold_percentile == percentile


def test_threshold_equals_percentile_of_training_mse(normal_df, monkeypatch):
    """Threshold must be precisely np.percentile(training_mse, p) from the trained model."""
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg(75))
    model = AutoencoderModel(input_dim=3)
    X = normal_df.values
    model.train(X)

    X_scaled = model.scaler.transform(X)
    recon = model.model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - recon), axis=1)
    expected = float(np.percentile(mse, 75))

    assert model.threshold == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_save_and_load_restores_threshold(normal_df, tmp_path, monkeypatch):
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg())
    model = AutoencoderModel(input_dim=3)
    model.train(normal_df.values)
    original_threshold = model.threshold
    model.save(str(tmp_path / "ae"), metrics={"mse_mean": 0.1})

    loaded = AutoencoderModel()
    loaded.load(str(tmp_path / "ae"))

    assert loaded.threshold == pytest.approx(original_threshold, rel=1e-5)
    assert loaded.model is not None
    assert loaded.scaler is not None


def test_load_restores_input_dim(normal_df, tmp_path, monkeypatch):
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg())
    model = AutoencoderModel(input_dim=3)
    model.train(normal_df.values)
    model.save(str(tmp_path / "ae"))

    loaded = AutoencoderModel()
    loaded.load(str(tmp_path / "ae"))
    assert loaded.input_dim == 3


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_contains_architecture_and_threshold_percentile(normal_df, tmp_path, monkeypatch):
    monkeypatch.setattr("models.autoencoder.get_config", lambda: _cfg(75))
    model = AutoencoderModel(input_dim=3)
    model.train(normal_df.values)
    model.save(str(tmp_path / "ae"))
    meta = model.get_metadata(str(tmp_path / "ae"))

    assert meta["model_type"] == "autoencoder"
    assert meta["threshold_percentile"] == 75
    assert isinstance(meta["architecture"], list)
    assert meta["architecture"][0] == 3
    assert meta["n_training_samples"] == len(normal_df)

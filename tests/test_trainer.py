# tests/test_trainer.py

import json
import os
from unittest.mock import patch

import pytest

from models.trainer import (
    cross_validate_model,
    train_autoencoder,
    train_random_forest,
    train_svm,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _ae_cfg(train_on_anomalous=False):
    return {"training": {
        "ae_train_on_anomalous": train_on_anomalous,
        "ae_epochs": 2,
        "ae_patience": 1,
        "ae_batch_size": 32,
        "ae_threshold_percentile": 95,
        "use_gpu": False,
    }}


def _patch_ae(monkeypatch, train_on_anomalous=False):
    cfg = _ae_cfg(train_on_anomalous)
    monkeypatch.setattr("models.trainer.get_config", lambda: cfg)
    monkeypatch.setattr("models.autoencoder.get_config", lambda: cfg)


def _rf_cfg():
    return {"training": {"n_estimators": 10, "use_gpu": False}}


def _svm_cfg(kernel="linear"):
    return {"training": {"svm_kernel": kernel, "svm_max_iter": 500, "use_gpu": False}}


# ---------------------------------------------------------------------------
# Basic train functions
# ---------------------------------------------------------------------------

def test_train_autoencoder_on_unlabelled_data(normal_csv, tmp_path, monkeypatch):
    _patch_ae(monkeypatch)
    metrics = train_autoencoder(normal_csv, output_path=str(tmp_path / "ae"))
    assert isinstance(metrics, dict)
    assert metrics.get("mse_mean", 0) > 0


def test_train_random_forest_on_labelled_data(labelled_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("models.trainer.get_config", lambda: _rf_cfg())
    monkeypatch.setattr("models.random_forest.get_config", lambda: _rf_cfg())
    metrics = train_random_forest(labelled_csv, output_path=str(tmp_path / "rf"))
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics


def test_train_svm_on_labelled_data(labelled_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("models.trainer.get_config", lambda: _svm_cfg())
    monkeypatch.setattr("models.svm.get_config", lambda: _svm_cfg())
    metrics = train_svm(labelled_csv, output_path=str(tmp_path / "svm"))
    assert isinstance(metrics, dict)
    assert "precision" in metrics


def test_train_autoencoder_returns_none_for_missing_file(tmp_path, monkeypatch):
    _patch_ae(monkeypatch)
    result = train_autoencoder(str(tmp_path / "nonexistent.csv"))
    assert result is None


def test_train_random_forest_returns_none_without_label_column(normal_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("models.trainer.get_config", lambda: _rf_cfg())
    result = train_random_forest(normal_csv, output_path=str(tmp_path / "rf"))
    assert result is None


# ---------------------------------------------------------------------------
# Anomalous-row filtering
# ---------------------------------------------------------------------------

def test_train_autoencoder_warns_when_anomalous_rows_present(labelled_csv, tmp_path, monkeypatch):
    _patch_ae(monkeypatch, train_on_anomalous=False)
    with patch("models.trainer.logger") as mock_logger:
        train_autoencoder(labelled_csv, output_path=str(tmp_path / "ae"))
    assert mock_logger.warning.called
    assert "anomalous" in str(mock_logger.warning.call_args_list).lower()


def test_train_autoencoder_no_anomalous_warning_when_flag_enabled(labelled_csv, tmp_path, monkeypatch):
    _patch_ae(monkeypatch, train_on_anomalous=True)
    with patch("models.trainer.logger") as mock_logger:
        train_autoencoder(labelled_csv, output_path=str(tmp_path / "ae"))
    anomalous_warnings = [
        c for c in mock_logger.warning.call_args_list
        if "anomalous" in str(c).lower()
    ]
    assert len(anomalous_warnings) == 0


def test_train_autoencoder_filtering_reduces_sample_count(labelled_csv, tmp_path, monkeypatch):
    """Filtering anomalous rows must produce a model trained on fewer samples."""
    _patch_ae(monkeypatch, train_on_anomalous=False)
    train_autoencoder(labelled_csv, output_path=str(tmp_path / "filtered"))

    _patch_ae(monkeypatch, train_on_anomalous=True)
    train_autoencoder(labelled_csv, output_path=str(tmp_path / "unfiltered"))

    with open(tmp_path / "filtered" / "metadata.json") as f:
        n_filtered = json.load(f)["n_training_samples"]
    with open(tmp_path / "unfiltered" / "metadata.json") as f:
        n_unfiltered = json.load(f)["n_training_samples"]

    assert n_filtered < n_unfiltered


# ---------------------------------------------------------------------------
# Attribute assignment
# ---------------------------------------------------------------------------

def test_training_dataset_set_for_autoencoder(labelled_csv, tmp_path, monkeypatch):
    _patch_ae(monkeypatch)
    train_autoencoder(labelled_csv, output_path=str(tmp_path / "ae"))
    with open(tmp_path / "ae" / "metadata.json") as f:
        meta = json.load(f)
    assert meta["training_dataset"] == os.path.abspath(labelled_csv)


def test_training_dataset_set_for_random_forest(labelled_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("models.trainer.get_config", lambda: _rf_cfg())
    monkeypatch.setattr("models.random_forest.get_config", lambda: _rf_cfg())
    train_random_forest(labelled_csv, output_path=str(tmp_path / "rf"))
    with open(tmp_path / "rf" / "metadata.json") as f:
        meta = json.load(f)
    assert meta["training_dataset"] == os.path.abspath(labelled_csv)


def test_training_dataset_set_for_svm(labelled_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("models.trainer.get_config", lambda: _svm_cfg())
    monkeypatch.setattr("models.svm.get_config", lambda: _svm_cfg())
    train_svm(labelled_csv, output_path=str(tmp_path / "svm"))
    with open(tmp_path / "svm" / "metadata.json") as f:
        meta = json.load(f)
    assert meta["training_dataset"] == os.path.abspath(labelled_csv)


def test_feature_names_stored_in_rf_metadata(labelled_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("models.trainer.get_config", lambda: _rf_cfg())
    monkeypatch.setattr("models.random_forest.get_config", lambda: _rf_cfg())
    train_random_forest(labelled_csv, output_path=str(tmp_path / "rf"))
    with open(tmp_path / "rf" / "metadata.json") as f:
        meta = json.load(f)
    feature_names = [feat["feature"] for feat in meta["top_features"]]
    assert all(name in ("f1", "f2", "f3") for name in feature_names)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def test_cross_validate_autoencoder_returns_mse_metrics(labelled_csv, monkeypatch):
    ae_cfg = _ae_cfg()
    monkeypatch.setattr("models.autoencoder.get_config", lambda: ae_cfg)
    results = cross_validate_model("autoencoder", labelled_csv, k=3)

    assert results.get("model_type") == "autoencoder"
    assert results.get("k_folds") == 3
    assert "mse_mean" in results
    assert "mean" in results["mse_mean"]
    assert "std" in results["mse_mean"]


def test_cross_validate_random_forest_returns_classification_metrics(labelled_csv, monkeypatch):
    monkeypatch.setattr("models.random_forest.get_config", lambda: _rf_cfg())
    results = cross_validate_model("random_forest", labelled_csv, k=3)

    assert results.get("model_type") == "random_forest"
    for metric in ("accuracy", "f1_score", "precision", "recall"):
        assert metric in results
        assert 0.0 <= results[metric]["mean"] <= 1.0


def test_cross_validate_svm_returns_classification_metrics(labelled_csv, monkeypatch):
    monkeypatch.setattr("models.svm.get_config", lambda: _svm_cfg())
    results = cross_validate_model("svm", labelled_csv, k=3)

    assert results.get("model_type") == "svm"
    for metric in ("accuracy", "f1_score"):
        assert metric in results
        assert 0.0 <= results[metric]["mean"] <= 1.0


def test_cross_validate_requires_label_column_for_supervised_models(normal_csv):
    results = cross_validate_model("random_forest", normal_csv, k=3)
    assert results == {}


def test_cross_validate_returns_empty_for_missing_file(tmp_path):
    results = cross_validate_model("random_forest", str(tmp_path / "nope.csv"), k=3)
    assert results == {}

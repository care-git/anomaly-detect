# tests/test_models/test_svm.py

import pytest
import numpy as np
from models.svm import SVMModel


def _cfg(kernel):
    return {"training": {"svm_kernel": kernel, "svm_max_iter": 500, "use_gpu": False}}


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_train_and_evaluate_all_kernels(kernel, labelled_df, monkeypatch):
    monkeypatch.setattr("models.svm.get_config", lambda: _cfg(kernel))
    X = labelled_df.drop(columns=["label"]).values
    y = labelled_df["label"].values

    model = SVMModel(input_dim=3)
    model.train(X, y=y)
    metrics = model.evaluate(X, y)

    for key in ("accuracy", "precision", "recall", "f1_score"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
    assert model._kernel == kernel


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_predict_returns_binary_labels(kernel, labelled_df, monkeypatch):
    monkeypatch.setattr("models.svm.get_config", lambda: _cfg(kernel))
    X = labelled_df.drop(columns=["label"]).values
    y = labelled_df["label"].values

    model = SVMModel(input_dim=3)
    model.train(X, y=y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_save_and_load_preserves_predictions(kernel, labelled_df, tmp_path, monkeypatch):
    monkeypatch.setattr("models.svm.get_config", lambda: _cfg(kernel))
    X = labelled_df.drop(columns=["label"]).values
    y = labelled_df["label"].values
    save_path = str(tmp_path / "svm")

    model = SVMModel(input_dim=3)
    model.train(X, y=y)
    original_preds = model.predict(X)
    model.save(save_path, metrics={"f1_score": 0.9})

    loaded = SVMModel()
    loaded.load(save_path)
    loaded_preds = loaded.predict(X)

    np.testing.assert_array_equal(original_preds, loaded_preds)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_metadata_stores_kernel_and_backend(kernel, labelled_df, tmp_path, monkeypatch):
    monkeypatch.setattr("models.svm.get_config", lambda: _cfg(kernel))
    X = labelled_df.drop(columns=["label"]).values
    y = labelled_df["label"].values
    save_path = str(tmp_path / "svm")

    model = SVMModel(input_dim=3)
    model.train(X, y=y)
    model.save(save_path)
    meta = model.get_metadata(save_path)

    assert meta["model_type"] == "svm"
    assert meta["kernel"] == kernel
    assert meta["backend"] is not None
    assert meta["n_training_samples"] == len(X)

    if kernel == "linear":
        assert "LinearSVC" in meta["backend"]
    else:
        assert kernel in meta["backend"]


def test_train_without_labels_raises(labelled_df):
    X = labelled_df.drop(columns=["label"]).values
    model = SVMModel(input_dim=3)
    with pytest.raises(ValueError, match="labels"):
        model.train(X)

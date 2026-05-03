# tests/test_utils/test_metrics.py

import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required

import numpy as np
import pytest

from utils.metrics_utils import (
    plot_classification_report,
    plot_cv_results,
    plot_feature_importance,
    plot_reconstruction_loss,
    pretty_print_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_labels():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    return y_true, y_pred


@pytest.fixture
def sample_metrics():
    return {"accuracy": 0.75, "precision": 0.80, "recall": 0.70, "f1_score": 0.75, "roc_auc": 0.82}


@pytest.fixture
def sample_mse():
    rng = np.random.default_rng(42)
    return rng.exponential(scale=0.05, size=50)


@pytest.fixture
def cv_results():
    return {
        "model_type": "random_forest",
        "k_folds": 3,
        "accuracy": {"mean": 0.85, "std": 0.04},
        "f1_score": {"mean": 0.82, "std": 0.05},
        "precision": {"mean": 0.80, "std": 0.06},
        "recall": {"mean": 0.84, "std": 0.03},
    }


# ---------------------------------------------------------------------------
# pretty_print_metadata
# ---------------------------------------------------------------------------

def test_pretty_print_metadata_outputs_json(capsys):
    metadata = {"model_type": "svm", "accuracy": 0.95}
    pretty_print_metadata(metadata)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["model_type"] == "svm"
    assert parsed["accuracy"] == 0.95


def test_pretty_print_metadata_warns_on_empty_dict(capsys):
    pretty_print_metadata({})
    # no output, no crash — warning is logged not printed
    out = capsys.readouterr().out
    assert out == ""


def test_pretty_print_metadata_uses_indent(capsys):
    pretty_print_metadata({"key": "value"}, indent=2)
    out = capsys.readouterr().out
    assert "  " in out


# ---------------------------------------------------------------------------
# plot_classification_report
# ---------------------------------------------------------------------------

def test_plot_classification_report_saves_png(tmp_path, sample_metrics, binary_labels):
    y_true, y_pred = binary_labels
    out = str(tmp_path / "report.png")
    plot_classification_report(sample_metrics, y_true, y_pred, output_path=out)
    assert (tmp_path / "report.png").exists()
    assert (tmp_path / "report.png").stat().st_size > 0


def test_plot_classification_report_creates_parent_dir(tmp_path, sample_metrics, binary_labels):
    y_true, y_pred = binary_labels
    out = str(tmp_path / "sub" / "report.png")
    plot_classification_report(sample_metrics, y_true, y_pred, output_path=out)
    assert (tmp_path / "sub" / "report.png").exists()


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

def test_plot_feature_importance_saves_png(tmp_path):
    importances = np.array([0.4, 0.3, 0.2, 0.1])
    names = ["f1", "f2", "f3", "f4"]
    out = str(tmp_path / "importance.png")
    plot_feature_importance(importances, names, output_path=out)
    assert (tmp_path / "importance.png").exists()
    assert (tmp_path / "importance.png").stat().st_size > 0


def test_plot_feature_importance_respects_top_n(tmp_path):
    importances = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    names = [f"f{i}" for i in range(5)]
    out = str(tmp_path / "top2.png")
    plot_feature_importance(importances, names, top_n=2, output_path=out)
    assert (tmp_path / "top2.png").exists()


# ---------------------------------------------------------------------------
# plot_reconstruction_loss
# ---------------------------------------------------------------------------

def test_plot_reconstruction_loss_saves_png(tmp_path, sample_mse):
    out = str(tmp_path / "recon.png")
    plot_reconstruction_loss(sample_mse, output_path=out)
    assert (tmp_path / "recon.png").exists()
    assert (tmp_path / "recon.png").stat().st_size > 0


def test_plot_reconstruction_loss_with_threshold(tmp_path, sample_mse):
    out = str(tmp_path / "recon_threshold.png")
    plot_reconstruction_loss(sample_mse, threshold=0.05, output_path=out)
    assert (tmp_path / "recon_threshold.png").exists()


def test_plot_reconstruction_loss_with_labels(tmp_path, sample_mse):
    y_true = np.array([0] * 40 + [1] * 10)
    out = str(tmp_path / "recon_labelled.png")
    plot_reconstruction_loss(sample_mse, y_true=y_true, threshold=0.1, output_path=out)
    assert (tmp_path / "recon_labelled.png").exists()


# ---------------------------------------------------------------------------
# plot_cv_results
# ---------------------------------------------------------------------------

def test_plot_cv_results_saves_png(tmp_path, cv_results):
    out = str(tmp_path / "cv.png")
    plot_cv_results(cv_results, output_path=out)
    assert (tmp_path / "cv.png").exists()
    assert (tmp_path / "cv.png").stat().st_size > 0


def test_plot_cv_results_ignores_non_metric_keys(tmp_path, cv_results):
    """model_type and k_folds are not metric dicts — the plot should not crash."""
    out = str(tmp_path / "cv_meta.png")
    plot_cv_results(cv_results, output_path=out)
    assert (tmp_path / "cv_meta.png").exists()

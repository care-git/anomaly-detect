# tests/test_models/test_benchmark.py

import json

import pandas as pd

from models.benchmark import COMPARISON_COLS, benchmark_models


def _patch_fast(monkeypatch):
    """Patch all three model configs to use minimal hyperparameters."""
    ae_cfg = {"training": {"ae_epochs": 2, "ae_patience": 1, "ae_batch_size": 32, "ae_threshold_percentile": 95}}
    rf_cfg = {"training": {"n_estimators": 10, "use_gpu": False}}
    svm_cfg = {"training": {"svm_kernel": "linear", "svm_max_iter": 500, "use_gpu": False}}
    monkeypatch.setattr("models.autoencoder.get_config", lambda: ae_cfg)
    monkeypatch.setattr("models.random_forest.get_config", lambda: rf_cfg)
    monkeypatch.setattr("models.svm.get_config", lambda: svm_cfg)


# ---------------------------------------------------------------------------
# Return value structure
# ---------------------------------------------------------------------------

def test_benchmark_returns_dataframe(labelled_csv, monkeypatch):
    _patch_fast(monkeypatch)
    result = benchmark_models(labelled_csv)
    assert isinstance(result, pd.DataFrame)


def test_benchmark_has_one_row_per_model(labelled_csv, monkeypatch):
    _patch_fast(monkeypatch)
    result = benchmark_models(labelled_csv)
    assert set(result.index) == {"autoencoder", "random_forest", "svm"}


def test_benchmark_columns_match_comparison_cols(labelled_csv, monkeypatch):
    """No AE-specific reconstruction metrics (mse_mean, mae_mean etc.) should leak into the table."""
    _patch_fast(monkeypatch)
    result = benchmark_models(labelled_csv)
    for col in result.columns:
        assert col in COMPARISON_COLS, f"Unexpected column in benchmark table: {col!r}"


def test_benchmark_metrics_are_in_valid_range(labelled_csv, monkeypatch):
    _patch_fast(monkeypatch)
    result = benchmark_models(labelled_csv)
    for col in COMPARISON_COLS:
        if col in result.columns:
            for val in result[col].dropna():
                assert 0.0 <= float(val) <= 1.0, f"{col}={val} out of [0, 1]"


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def test_benchmark_saves_csv_when_output_dir_provided(labelled_csv, tmp_path, monkeypatch):
    _patch_fast(monkeypatch)
    benchmark_models(labelled_csv, output_dir=str(tmp_path))
    assert (tmp_path / "benchmark_results.csv").exists()


def test_benchmark_saves_json_when_output_dir_provided(labelled_csv, tmp_path, monkeypatch):
    _patch_fast(monkeypatch)
    benchmark_models(labelled_csv, output_dir=str(tmp_path))
    json_path = tmp_path / "benchmark_results.json"
    assert json_path.exists()
    with open(json_path) as f:
        records = json.load(f)
    assert isinstance(records, list)
    assert len(records) == 3


def test_benchmark_csv_columns_match_comparison_cols(labelled_csv, tmp_path, monkeypatch):
    _patch_fast(monkeypatch)
    benchmark_models(labelled_csv, output_dir=str(tmp_path))
    saved = pd.read_csv(tmp_path / "benchmark_results.csv", index_col="model")
    for col in saved.columns:
        assert col in COMPARISON_COLS, f"Unexpected column in saved CSV: {col!r}"


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------

def test_benchmark_returns_empty_df_for_missing_file(tmp_path, monkeypatch):
    _patch_fast(monkeypatch)
    result = benchmark_models(str(tmp_path / "nope.csv"))
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_benchmark_returns_empty_df_without_label_column(normal_csv, monkeypatch):
    _patch_fast(monkeypatch)
    result = benchmark_models(normal_csv)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# AE trains on normal-only data
# ---------------------------------------------------------------------------

def test_benchmark_ae_trains_on_normal_only(labelled_csv, monkeypatch):
    """
    AutoencoderModel.train must be called with data that contains only normal samples.

    The labelled fixture has normal data near 0 and anomalous data near 3. If AE
    were trained on mixed data the maximum feature value in its training set would
    approach 3; normal-only training keeps it well below 1.5.
    """
    from models.autoencoder import AutoencoderModel

    captured = []
    original_train = AutoencoderModel.train

    def spy_train(self, X, **kwargs):
        captured.append(X.copy())
        return original_train(self, X, **kwargs)

    _patch_fast(monkeypatch)
    monkeypatch.setattr(AutoencoderModel, "train", spy_train)
    benchmark_models(labelled_csv)

    assert len(captured) >= 1
    ae_train_X = captured[0]
    # Normal samples cluster near 0; anomalous near 3. Mixed training would
    # produce a mean > 1 across features.
    assert ae_train_X.mean() < 1.5

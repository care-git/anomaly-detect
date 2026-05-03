# tests/test_cli/test_main.py

import sys
import pytest

from cli.main import main, parse_args


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_config_state():
    """Prevent set_config_path calls from polluting the module cache."""
    import utils.config_loader as cl
    original_path = cl._config_path
    original_cache = cl._config_cache
    yield
    cl._config_path = original_path
    cl._config_cache = original_cache


# ---------------------------------------------------------------------------
# --version / --help
# ---------------------------------------------------------------------------

def test_version_flag_exits_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_version_output_contains_version_string(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "--version"])
    with pytest.raises(SystemExit):
        main()
    out = capsys.readouterr().out
    assert "anomaly-detect" in out


def test_help_flag_exits_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


@pytest.mark.parametrize("subcommand", ["capture", "preprocess", "dataset", "train", "detect", "benchmark"])
def test_subcommand_help_exits_zero(subcommand, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", subcommand, "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Argument parsing — global options
# ---------------------------------------------------------------------------

def test_parse_args_config_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "train", "--model", "autoencoder", "--input", "data.csv"])
    args = parse_args()
    assert args.config == "config/config.yml"


def test_parse_args_config_custom(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "--config", "custom/config.yml", "train", "--model", "autoencoder", "--input", "data.csv"])
    args = parse_args()
    assert args.config == "custom/config.yml"


# ---------------------------------------------------------------------------
# Argument parsing — train subcommand
# ---------------------------------------------------------------------------

def test_parse_args_train_model_and_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "train", "--model", "random_forest", "--input", "data.csv"])
    args = parse_args()
    assert args.command == "train"
    assert args.model == "random_forest"
    assert args.input == "data.csv"


def test_parse_args_train_cv_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "train", "--model", "svm", "--input", "data.csv", "--cv", "--cv-folds", "10"])
    args = parse_args()
    assert args.cv is True
    assert args.cv_folds == 10


def test_parse_args_train_cv_default_folds(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "train", "--model", "autoencoder", "--input", "data.csv", "--cv"])
    args = parse_args()
    assert args.cv_folds == 5


def test_parse_args_train_output_is_optional(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "train", "--model", "autoencoder", "--input", "data.csv"])
    args = parse_args()
    assert args.output is None


# ---------------------------------------------------------------------------
# Argument parsing — detect subcommand
# ---------------------------------------------------------------------------

def test_parse_args_detect_file_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "detect", "--model", "autoencoder", "--model-path", "/tmp/m", "--input", "capture.pcap", "--output", "results.csv"])
    args = parse_args()
    assert args.command == "detect"
    assert args.model == "autoencoder"
    assert args.input == "capture.pcap"
    assert args.live is False


def test_parse_args_detect_live_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "detect", "--model", "random_forest", "--model-path", "/tmp/m", "--live", "--interface", "eth0"])
    args = parse_args()
    assert args.live is True
    assert args.interface == "eth0"


# ---------------------------------------------------------------------------
# Argument parsing — benchmark subcommand
# ---------------------------------------------------------------------------

def test_parse_args_benchmark_requires_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "benchmark"])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code != 0


def test_parse_args_benchmark_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "benchmark", "--input", "dataset.csv"])
    args = parse_args()
    assert args.command == "benchmark"
    assert args.input == "dataset.csv"


# ---------------------------------------------------------------------------
# Argument parsing — capture / preprocess / dataset
# ---------------------------------------------------------------------------

def test_parse_args_capture_interface_and_duration(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "capture", "--interface", "eth0", "--duration", "30"])
    args = parse_args()
    assert args.command == "capture"
    assert args.interface == "eth0"
    assert args.duration == 30


def test_parse_args_preprocess_label(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "preprocess", "--input", "cap.pcap", "--label", "1"])
    args = parse_args()
    assert args.label == 1


def test_parse_args_dataset_combine(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "dataset", "--combine", "a.csv", "b.csv"])
    args = parse_args()
    assert args.combine == ["a.csv", "b.csv"]


# ---------------------------------------------------------------------------
# Dispatch — main() routes to the correct handler
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcommand,dispatcher_path,extra_argv", [
    ("train",     "cli.main.run_train_model",   ["--model", "autoencoder", "--input", "data.csv"]),
    ("detect",    "cli.main.run_detection",     ["--model", "autoencoder", "--model-path", "/tmp/m", "--input", "cap.pcap"]),
    ("benchmark", "cli.main.run_benchmark",     ["--input", "data.csv"]),
    ("preprocess","cli.main.run_preprocessor",  ["--input", "cap.pcap"]),
    ("capture",   "cli.main.run_capture",       ["--interface", "eth0"]),
    ("dataset",   "cli.main.run_dataset_utils", ["--combine", "a.csv"]),
])
def test_main_dispatches_to_correct_handler(subcommand, dispatcher_path, extra_argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", subcommand] + extra_argv)
    monkeypatch.setattr("cli.main.set_config_path", lambda p: None)

    called_with = []
    monkeypatch.setattr(dispatcher_path, lambda args: called_with.append(args))

    main()

    assert len(called_with) == 1


# ---------------------------------------------------------------------------
# --config wiring
# ---------------------------------------------------------------------------

def test_main_passes_custom_config_to_set_config_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "--config", "custom/config.yml", "train", "--model", "autoencoder", "--input", "data.csv"])
    monkeypatch.setattr("cli.main.run_train_model", lambda args: None)

    applied = []
    monkeypatch.setattr("cli.main.set_config_path", lambda p: applied.append(p))

    main()

    assert "custom/config.yml" in applied


def test_main_passes_default_config_path_when_not_overridden(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["anomaly-detect", "train", "--model", "autoencoder", "--input", "data.csv"])
    monkeypatch.setattr("cli.main.run_train_model", lambda args: None)

    applied = []
    monkeypatch.setattr("cli.main.set_config_path", lambda p: applied.append(p))

    main()

    assert applied[0] == "config/config.yml"

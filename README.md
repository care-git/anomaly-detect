# Anomaly Detect

A modular, machine learning-based threat detection pipeline for identifying anomalies in network traffic.

Designed for use by security analysts, researchers, and engineers needing a flexible framework for real-time or file-based detection of network threats.

Most machine learning-based intrusion detection research achieves strong benchmark results and stops there. The gap between a well-evaluated model and a system that captures live traffic, handles real-world datasets, integrates with operational tooling, and can be extended without rewriting the core logic is rarely addressed, as most academic implementations were never designed with deployment in mind.

This project was built to close that gap. Detection accuracy matters, but so does modularity, configurability, and SIEM integration. This pipeline treats these as first-class requirements alongside model performance, and the result is a framework that security engineers and researchers can actually run, configure, and build on, rather than one that exists solely to reproduce benchmark figures.

---

## Features

- **Three detection models** - Autoencoder (unsupervised), Random Forest (supervised), and SVM (supervised)
- **Live and offline packet capture** via Scapy with ~30 features extracted per packet
- **Benchmark mode** - trains all three models on the same split and produces a side-by-side comparison
- **k-fold cross-validation** for robust evaluation of any model
- **Visual evaluation** - classification report charts, feature importance, reconstruction loss distribution, CV results
- **ROC-AUC, F1, precision, recall** metrics across all models; MSE/MAE reconstruction metrics for the Autoencoder
- **GPU acceleration** - TensorFlow uses Metal (Apple Silicon) or CUDA (Linux/WSL2); RF and SVM can use RAPIDS cuML on Linux + NVIDIA. Windows native runs CPU-only - use WSL2 for full GPU support
- **Wazuh SIEM integration** - alert forwarding via rotating log file, syslog UDP, or both
- **Unified CLI** with per-command help, `--config` override, and `--version`
- Fully customisable via YAML config
- Pytest-based unit test suite

---

## Installation

### 1. Clone the repository

This project uses [Git LFS](https://git-lfs.github.com/) for compiled public datasets. Install LFS before cloning.

```bash
git lfs install

git clone https://github.com/care-git/anomaly-detect.git
cd anomaly-detect
git lfs pull
```

### 2. Create the environment

Use the file that matches your platform:

**macOS - Apple Silicon (M1/M2/M3)**
```bash
conda env create -f environment-mac.yml
conda activate anomaly-detect
pip install -e .
```

**Linux - x86_64**
```bash
conda env create -f environment-linux.yml
conda activate anomaly-detect
pip install -e .
```

**Windows 10/11 - x86_64**
```bash
conda env create -f environment-windows.yml
conda activate anomaly-detect
pip install -e .
```

> **Packet capture on Windows** requires [Npcap](https://npcap.com) to be installed before running any `anomaly-detect capture` commands. Install it via winget or download the installer manually from the link above:
> ```powershell
> winget install --id Npcap.Npcap
> ```
> Network interface names differ from Linux/macOS - use `"Wi-Fi"` or `"Ethernet"` (check Device Manager or run `getmac /v`). Update `interface` in `config/config.yml` accordingly.

**Windows - WSL2 (recommended for NVIDIA GPU users)**

WSL2 runs a real Linux kernel inside Windows and gives full CUDA and cuML support. Install WSL2, enable virtualisation in PowerShell, and enable virtualisation in BIOS. Then follow the Linux installation instructions below within your WSL2 terminal:

```powershell
wsl --install
wsl --set-default-version 2

# enable WSL2 virtualisation (requires reboot - may also require enabling within BIOS)
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart  
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

wsl --install -d Ubuntu
```

```bash
# Inside WSL2 terminal
conda env create -f environment-linux.yml
conda activate anomaly-detect
pip install -e .
```

> **Packet capture in WSL2** has caveats. WSL2 uses a virtualised network adapter separate from the Windows host by default. On Windows 11 22H2 or later you can enable mirrored networking (`[wsl2] networkingMode=mirrored` in `.wslconfig`) to share the host interface, but results vary by hardware. If capture is a requirement, use the native Windows installation instead, or some combination of WSL2 and native Windows.

**pip only (no conda)**
```bash
pip install -e .
```

### 3. GPU acceleration (optional)

**Apple Silicon** - `tensorflow-metal` is included in `environment-mac.yml` and enables Metal GPU acceleration for the Autoencoder automatically. No config change needed. The `use_gpu` flag in `config/config.yml` controls cuML (RF/SVM only) and should be left `false` on macOS.

**Linux + NVIDIA GPU (RAPIDS cuML)** - cuML is already included in `environment-linux.yml`. Recreate the environment if needed, then enable GPU in config:

```bash
conda env create -f environment-linux.yml
conda activate anomaly-detect
```

Set `use_gpu: true` in `config/config.yml`. Random Forest and SVM training will use GPU-accelerated cuML backends automatically. The pipeline falls back to sklearn silently if cuML is not found.

**Windows native** - `environment-windows.yml` installs `tensorflow-cpu`. All three models train on CPU. No config change is needed; `use_gpu` has no effect on native Windows. For GPU acceleration, use WSL2 virtualisation.

**Windows + WSL2 + NVIDIA GPU** - follow the Linux cuML instructions above inside your WSL2 terminal. NVIDIA's CUDA drivers for WSL2 are installed at the Windows host level; no separate CUDA install is needed inside WSL2.

---

## Configuration

All pipeline settings are controlled by `config/config.yml`. Key options:

| Section | Key | Description |
|---|---|---|
| `capture` | `interface` | Network interface for live capture |
| `capture` | `duration` / `packet_count` | Capture stop conditions |
| `training` | `model_type` | Default model (`autoencoder`, `random_forest`, `svm`) |
| `training` | `n_estimators` | Number of trees for Random Forest |
| `training` | `svm_kernel` | SVM kernel (`linear`, `rbf`, `poly`) |
| `training` | `svm_max_iter` | Max solver iterations for LinearSVC |
| `training` | `ae_epochs` / `ae_patience` / `ae_batch_size` | Autoencoder training hyperparameters |
| `training` | `use_gpu` | Enable GPU acceleration (`true`/`false`) |
| `siem` | `mode` | Alert output mode (`file`, `syslog`, `both`) |
| `siem` | `log_max_bytes` / `log_backup_count` | Rotating log file limits |

Pass an alternate config at runtime with `--config`:
```bash
anomaly-detect --config config/custom.yml train --model svm
```

---

## Usage

```bash
anomaly-detect --help
anomaly-detect <subcommand> --help
```

### Capture

Capture live network traffic to a PCAP file.

```bash
anomaly-detect capture --interface eth0 --duration 30 --output data/captures/sample.pcap
```

### Preprocess

Extract packet features from a PCAP file to a labelled CSV.

```bash
anomaly-detect preprocess --input data/captures/sample.pcap --output data/processed/sample.csv

# Attach a ground-truth label column (1 = attack, 0 = normal)
anomaly-detect preprocess --input data/captures/attack.pcap --output data/processed/attack.csv --label 1
```

### Dataset

Combine, split, and balance labelled CSV datasets.

```bash
anomaly-detect dataset --combine data/processed/normal.csv data/processed/attack.csv --output data/combined/dataset.csv
anomaly-detect dataset --balance data/combined/dataset.csv
```

### Train

Train a model on a preprocessed CSV. Saves the model, scaler, metadata, and evaluation plots to the output directory.

```bash
# Single train/evaluate pass
anomaly-detect train --model autoencoder --input data/processed/sample.csv
anomaly-detect train --model random_forest --input data/combined/dataset.csv --output data/models/rf/
anomaly-detect train --model svm --input data/combined/dataset.csv

# k-fold cross-validation
anomaly-detect train --model random_forest --input data/combined/dataset.csv --cv
anomaly-detect train --model svm --input data/combined/dataset.csv --cv --cv-folds 10
```

### Detect

Run detection on a PCAP file or as a live packet monitor.

```bash
# File-based detection
anomaly-detect detect --model autoencoder \
  --model-path data/models/autoencoder/autoencoder_model \
  --input data/captures/sample.pcap \
  --output data/detection/results.csv

# Live detection
anomaly-detect detect --model random_forest \
  --model-path data/models/random_forest/random_forest_model \
  --live --interface eth0
```

### Benchmark

Train all three models on the same labelled dataset and compare evaluation metrics side by side.

```bash
anomaly-detect benchmark --input data/combined/dataset.csv --output data/models/benchmark/
```

---

## Project Structure

```
anomaly-detect/
├── cli/                      # CLI entry point and argument parsing
├── config/                   # YAML configuration (config.yml)
├── core/                     # Packet capture, preprocessing, dataset utilities
├── models/                   # Model classes, trainer, detector, benchmark
│   ├── autoencoder.py
│   ├── random_forest.py
│   ├── svm.py
│   ├── trainer.py            # train_* functions and cross_validate_model
│   ├── detector.py
│   └── benchmark.py
├── siem/                     # Wazuh alert forwarding
├── utils/                    # Config loader, logger, progress, GPU helpers, metrics
├── tests/                    # Pytest unit test suite
├── environment-mac.yml       # Conda environment - macOS Apple Silicon
├── environment-linux.yml     # Conda environment - Linux x86_64
├── environment-windows.yml   # Conda environment - Windows 10/11 (native)
└── data/                     # Gitignored - captures, processed CSVs, models, logs
```

---

## Testing

### Running the suite

Activate the conda environment for your platform before running tests. The suite relies on platform-specific packages (TensorFlow, tensorflow-metal, cuML) that are not available cross-platform.

```bash
# Full suite - recommended before any commit
python -m pytest tests/

# Verbose output (shows each test name)
python -m pytest tests/ -v

# Stop on first failure
python -m pytest tests/ -x

# Run tests matching a keyword
python -m pytest tests/ -k svm
python -m pytest tests/ -k autoencoder
```

### Test structure

```
tests/
├── conftest.py                  # Shared fixtures (labelled_df, normal_df, labelled_csv, normal_csv)
├── test_capture.py              # Packet capture and live monitor
├── test_dataset_utils.py        # Dataset combine / split / balance
├── test_detector.py             # File-based and live detection
├── test_preprocessor.py         # PCAP feature extraction
├── test_trainer.py              # Training, filtering, cross-validation for all 3 models
├── test_cli/
│   └── test_main.py             # CLI argument parsing and handler dispatch
├── test_models/
│   ├── test_autoencoder.py      # AE training, threshold, save/load, metadata
│   ├── test_benchmark.py        # Three-model benchmark, output files, normal-only AE
│   ├── test_random_forest.py    # RF training, feature importance, save/load
│   └── test_svm.py              # SVM all kernels (linear, rbf, poly), save/load
├── test_platform/
│   ├── test_macos.py            # macOS-specific: Metal GPU, ANSI, no WSL2 in logs
│   ├── test_linux.py            # Linux-specific: RAPIDS hint, eth0 config docs
│   ├── test_windows.py          # Windows-specific: ANSI ctypes, WSL2 hint, Npcap, CPU-only
│   └── test_wsl2.py             # WSL2-specific: detection signals, Linux paths, CUDA GPU
├── test_siem/
│   └── test_wazuh_forwarder.py  # File/syslog/both modes, dry_run, timestamps, failure paths
└── test_utils/
    ├── test_config_loader.py    # Default/custom path, caching, cache invalidation
    ├── test_gpu_utils.py        # cuml_available, setup_gpu idempotency, platform log branches
    ├── test_metrics.py          # pretty_print_metadata, all 4 plot functions save PNG
    └── test_progress.py         # tqdm_bar, _ansi_supported branches, spinner lifecycle
```

### Markers

Markers filter tests by platform or resource requirement. They are defined in `pyproject.toml` and can be combined.

| Marker | Meaning |
|---|---|
| `macos` | macOS (darwin) only - skipped on Linux and Windows |
| `linux` | Linux only - skipped on macOS and Windows |
| `windows` | Windows (win32) only - skipped on macOS and Linux |
| `wsl2` | WSL2 only - skipped unless kernel release contains `microsoft` or `WSL_DISTRO_NAME` is set |
| `gpu` | Requires a physical GPU - skipped if no devices found |
| `slow` | Long-running tests - omit with `-m "not slow"` for fast feedback loops |

```bash
# Run only the platform tests relevant to this machine
python -m pytest tests/test_platform/ -m macos       # on macOS
python -m pytest tests/test_platform/ -m linux       # on Linux / WSL2
python -m pytest tests/test_platform/ -m windows     # on Windows

# Run only GPU tests
python -m pytest -m gpu

# Exclude slow tests
python -m pytest -m "not slow"

# Combine: GPU tests that are not slow
python -m pytest -m "gpu and not slow"
```

### Per-platform commands

**macOS (Apple Silicon)**
```bash
conda activate anomaly-detect
python -m pytest tests/
# Platform tests for macOS run automatically; Linux/Windows/WSL2 files are skipped.
```

**Linux (native)**
```bash
conda activate anomaly-detect
python -m pytest tests/
# test_platform/test_linux.py runs; macOS/Windows/WSL2 files are skipped.
# cuML tests skip automatically if RAPIDS is not installed.
```

**Linux (WSL2 inside Windows)**
```bash
conda activate anomaly-detect
python -m pytest tests/
# test_platform/test_linux.py and test_platform/test_wsl2.py both run.
# test_platform/test_macos.py and test_platform/test_windows.py are skipped.
```

**Windows (native)**
```bash
conda activate anomaly-detect
python -m pytest tests/
# test_platform/test_windows.py runs; all others are skipped.
# GPU tests skip automatically if no DirectX 12 device is found.
```

### GPU tests

`@pytest.mark.gpu` tests are skipped automatically when no GPU is present - they do not require any manual flag. To explicitly run or exclude them:

```bash
python -m pytest -m gpu        # only GPU tests (skip if no device)
python -m pytest -m "not gpu"  # skip all GPU tests regardless
```

On **macOS**, GPU tests verify Metal device detection via `tensorflow-metal`. On **Linux/WSL2**, they verify CUDA passthrough. Native Windows has no GPU tests - the `@pytest.mark.gpu` tests are skipped on CPU-only installs.

### WSL2 packet capture note

WSL2 uses a virtualised network adapter separate from the Windows host. Packet capture tests that exercise Scapy against a real interface are skipped inside WSL2 by default. If you have enabled mirrored networking (`networkingMode=mirrored` in `.wslconfig` on Windows 11 22H2+), you can run capture tests manually against the mirrored interface.

---

## SIEM Integration

Anomalies are forwarded as structured JSON alerts in Wazuh-compatible format.

Configure output mode in `config/config.yml`:

```yaml
siem:
  mode: file      # Options: file, syslog, both
  log_path: data/logs/alerts.log
  log_max_bytes: 10485760     # 10 MB per file
  log_backup_count: 5
  syslog_host: 127.0.0.1
  syslog_port: 514
```

Point Wazuh's `ossec.conf` at the log file or configure a syslog input to receive alerts over UDP.

---

## Versioning

Version numbers are derived automatically from git tags using `setuptools-scm`. No manual version file editing is needed.

The version is accessible at runtime via:
```bash
anomaly-detect --version
```

**When running from source without a tag** (e.g. a fresh clone with no tags fetched), the version falls back to `1.1.0` as defined in `pyproject.toml`. To get the correct version on a new machine:
```bash
git fetch --tags
```

---

## License

MIT License - see [LICENSE](./LICENSE) for details.

---

## Author

Originally developed as a final year dissertation project by William Jecks.
BSc Cyber Security, De Montfort University.

# Anomaly Detect

A modular, machine learning-based threat detection pipeline for identifying anomalies in network traffic.

Designed for use by security analysts, researchers, and engineers needing a flexible framework for real-time or file-based detection of network threats.

---

## Features

- **Three detection models** - Autoencoder (unsupervised), Random Forest, and SVM (supervised)
- **Live and offline packet capture** via Scapy with ~30 features extracted per packet
- **Benchmark mode** - trains all three models on the same split and produces a side-by-side comparison
- **k-fold cross-validation** for robust evaluation of any model
- **Visual evaluation** - classification report charts, feature importance, reconstruction loss distribution, CV results
- **ROC-AUC, F1, precision, recall** metrics across all models; MSE/MAE reconstruction metrics for the Autoencoder
- **GPU acceleration** - TensorFlow uses Metal (Apple Silicon), CUDA (Linux), or DirectML (Windows); RF and SVM can use RAPIDS cuML on Linux + NVIDIA
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
git lfs pull          # download compiled dataset files
```

### 2. Create the environment

Use the file that matches your platform:

**macOS — Apple Silicon (M1/M2/M3)**
```bash
conda env create -f environment-mac.yml
conda activate anomaly-detect
pip install -e .
```

**Linux — x86_64**
```bash
conda env create -f environment-linux.yml
conda activate anomaly-detect
pip install -e .
```

**Windows 10/11 — native**
```bash
conda env create -f environment-windows.yml
conda activate anomaly-detect
pip install -e .
```

> **Packet capture on Windows** requires [Npcap](https://npcap.com) to be installed before running any `anomaly-detect capture` commands. Install it via winget or download the installer manually from npcap.com:
> ```powershell
> winget install --id Npcap.Npcap
> ```
> Network interface names differ from Linux/macOS — use `"Wi-Fi"` or `"Ethernet"` (check Device Manager or run `getmac /v`). Update `interface` in `config/config.yml` accordingly.

**Windows — WSL2 (recommended for NVIDIA GPU users)**

WSL2 runs a real Linux kernel inside Windows and gives full CUDA and cuML support. Install WSL2 and enable virtualisation in PowerShell, then follow the Linux installation instructions above inside your WSL2 terminal after restarting your PC:

```powershell
# In PowerShell (run as Administrator) — one-time WSL2 setup
wsl --install
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2    # optional - use WSL2 by default
```

```bash
# Inside WSL2 terminal — same as Linux
conda env create -f environment-linux.yml
conda activate anomaly-detect
pip install -e .
```

> **Packet capture in WSL2** has caveats. WSL2 uses a virtualised network adapter separate from the Windows host by default. On Windows 11 22H2 or later you can enable mirrored networking (`[wsl2] networkingMode=mirrored` in `.wslconfig`) to share the host interface, but results vary by hardware. If capture is a requirement, use the native Windows installation instead.

**pip only (no conda)**
```bash
pip install -e .
```

### 3. GPU acceleration (optional)

**Apple Silicon** — `tensorflow-metal` is included in `environment-mac.yml` and enables Metal GPU acceleration for the Autoencoder automatically. No config change needed. The `use_gpu` flag in `config/config.yml` controls cuML (RF/SVM only) and should be left `false` on macOS.

**Linux + NVIDIA GPU (RAPIDS cuML)** — cuML is already included in `environment-linux.yml`. Recreate the environment if needed, then enable GPU in config:

```bash
conda env create -f environment-linux.yml
conda activate anomaly-detect
```

Set `use_gpu: true` in `config/config.yml`. Random Forest and SVM training will use GPU-accelerated cuML backends automatically. The pipeline falls back to sklearn silently if cuML is not found.

**Windows + DirectML (Autoencoder only)** — DirectML enables DirectX 12 GPU acceleration for the Autoencoder on any modern NVIDIA, AMD, or Intel GPU without requiring CUDA. Uncomment `tensorflow-directml-plugin` in `environment-windows.yml`, recreate the environment, then enable GPU in config:

```bash
# In environment-windows.yml: uncomment `# - tensorflow-directml-plugin`
conda env create -f environment-windows.yml
conda activate anomaly-detect
```

Set `use_gpu: true` in `config/config.yml`. RF and SVM GPU acceleration is not available on Windows natively — use WSL2 for full cuML support.

**Windows + WSL2 + NVIDIA GPU (full GPU support)** — follow the Linux cuML instructions above inside your WSL2 terminal. NVIDIA's CUDA drivers for WSL2 are installed at the Windows host level; no separate CUDA install is needed inside WSL2.

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

# k-fold cross-validation (default 5 folds)
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

```bash
pytest tests/
pytest tests/ -v          # verbose
pytest tests/ -k svm      # run tests matching a keyword
```

---

## SIEM Integration

Anomalies are forwarded as structured JSON alerts in Wazuh-compatible format.

Configure output mode in `config/config.yml`:

```yaml
siem:
  mode: file          # Options: file, syslog, both
  log_path: data/logs/alerts.log
  log_max_bytes: 10485760    # 10 MB per file
  log_backup_count: 5        # keep 5 rotated backups
  syslog_host: 127.0.0.1
  syslog_port: 514
```

Point Wazuh's `ossec.conf` at the log file or configure a syslog input to receive alerts over UDP.

---

## Versioning

Version numbers are derived automatically from git tags using `setuptools-scm`. No manual version file editing is needed.

**To release a new version:**
```bash
git tag v1.2.5
git push origin v1.2.5
```

The version is then accessible at runtime via:
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

Originally developed by William Jecks as a final year dissertation project.
BSc Cyber Security, De Montfort University.

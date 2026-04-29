# Code Audit

## Bugs

**1. `test_autoencoder.py:44` — `AttributeError` crashes the save/load test**

`setUp` defines `self.output_path`, but `test_model_save_and_reload_works` uses `self.model_path` (which doesn't exist). This test raises `AttributeError` unconditionally and has never passed.

```python
# setUp
self.output_path = os.path.join(self.temp_dir.name, "ae_model")

# test (line 44) — wrong name
model.save(self.model_path, ...)  # AttributeError
```

Fix: rename `self.model_path` → `self.output_path` in the three test lines (44, 47, 52), or add `self.model_path = self.output_path` to `setUp`.

---

**2. `test_progress.py:6` — `ImportError` kills the entire test module**

The test imports `TqdmKerasCallback` from `utils/progress.py`, but that class is not defined there — it doesn't exist anywhere in the codebase. The whole test file fails to load.

```python
from utils.progress import tqdm_bar, single_bar, TqdmKerasCallback  # ImportError
```

Either implement `TqdmKerasCallback` in `progress.py` or remove the import and the test that uses it.

---

**3. `preprocessor.py:189` — label column named `"anomaly"`, trainers expect `"label"`**

When `--label` is passed to `preprocess`, the output column is `"anomaly"`. But `train_random_forest` and `train_svm` look for `"label"` and immediately return an error if it's missing. This breaks the natural `preprocess --label 1 → train --model random_forest` workflow.

```python
# preprocessor.py:189
df_batch["anomaly"] = label  # should be "label"
```

---

**4. `detector.py:51-60` — autoencoder runs inference twice**

`model.predict(X)` is called on line 51 (which internally runs the Keras model and computes MSE). Then for autoencoders, the detector immediately throws away the result and re-runs the same computation by reaching directly into `model.model.predict(...)` on line 55. The first call is pure waste, and the direct access to model internals breaks the `BaseModel` abstraction.

```python
predictions = model.predict(X)           # full inference here

if model_type == "autoencoder":
    reconstructions = model.model.predict(...)  # again, from scratch
    mse = ...
    predictions = (mse > threshold).astype(int) # overwrites previous result
```

The fix is to move this autoencoder-specific postprocessing into `AutoencoderModel.predict()` so the base `predict()` call is sufficient and the detector doesn't need to know the model's internals.

---

**5. `packet_utils.py:72` — bare `except:` swallows fatal signals**

```python
try:
    return float(x)
except:       # catches KeyboardInterrupt, SystemExit, MemoryError
    return x
```

Change to `except (TypeError, ValueError, OverflowError):`.

---

**6. `dataset_utils.py:84` — `test_size` parameter accepted but ignored**

`split_dataset(df, ..., train_size=0.8, test_size=0.2)` validates that they sum to 1, but `test_size` is never passed to `train_test_split`. sklearn computes it from `train_size` automatically, so behaviour is correct by accident, but the parameter is misleading.

---

**7. `dataset_utils.py:53,63` — cross-file deduplication never works**

`build_combined_dataset` adds a `source` column (the filename) to each DataFrame *before* calling `drop_duplicates()`, but only removes it *after*. Identical data rows from two different files get different `source` values, so they never deduplicate. The `drop_duplicates` call effectively only deduplicates within a single source file.

```python
df['source'] = os.path.basename(src)   # different per file
...
combined.drop_duplicates(inplace=True) # source column still present → no cross-file dedup
combined.drop(columns=['source'], ...)
```

Move `drop_duplicates` to after `drop(columns=['source'])`.

---

**8. `wazuh_forwarder.py:34` — mutates caller's dict in-place**

`alert['timestamp'] = ...` modifies the dict passed in by reference. The caller in `detector.py` constructed the alert dict and may reuse or log it. Standard practice is to work on a copy: `alert = {**alert, "timestamp": ...}`.

---

**9. `file_saver.py:127` — phantom path trick to create a directory**

```python
ensure_dir(os.path.join(base_dir, 'placeholder.tmp'))
```

This relies on `ensure_dir` calling `os.path.dirname()` internally to strip the filename — a roundabout way to just create `base_dir`. Replace with `os.makedirs(base_dir, exist_ok=True)`.

---

**10. `balance_labels` minority/majority variable assignment is inverted**

`value_counts()` returns in descending order, so `class_0` is the *majority* class — the opposite of what the variable name implies. The subsequent swap corrects the end result, but the code is actively misleading:

```python
class_0, class_1 = class_counts.index     # class_0 = majority, class_1 = minority
df_minority = df[df[label_col] == class_0] # labelled "minority" but is actually majority
df_majority = df[df[label_col] == class_1] # labelled "majority" but is actually minority
if len(df_majority) < len(df_minority):    # this is always True → always swaps
    df_minority, df_majority = df_majority, df_minority
```

The swap always executes, making the top assignment meaningless. Simplify to just find min/max directly.

---

## Design Issues

**11. Module-level `get_config()` at import time**

Almost every module runs `config = get_config()` at module level. This means importing `from models.detector import ...` in a test immediately tries to read `config/config.yml` relative to the CWD. If a test is run from any directory other than the project root, all imports fail. The config load should be deferred to function call time, or `CONFIG_PATH` should be resolved relative to `__file__`.

---

**12. `AutoencoderModel` doesn't call `super().__init__()`**

`BaseModel.__init__` sets `self.config` and `self.logger`. `AutoencoderModel.__init__` bypasses this entirely, using a module-level `get_logger()` call instead. This makes the base class contract meaningless for the autoencoder — it's the only model that doesn't honour it, even though all three models use module-level loggers and none use `self.logger` from the base class anyway. Either all models should call `super().__init__()` and use `self.logger`, or the base class logger setup should be removed.

---

**13. `wazuh_forwarder.py` module-level constants frozen at import**

`DEFAULT_MODE`, `DEFAULT_LOG_PATH`, etc. are evaluated once at import. This means the SIEM config is effectively immutable at runtime. If you want to test with `mode: syslog` but the module was already loaded with `mode: file`, you can't change it without reloading the module. These should be read inside `forward_alert()` each call (or at least not captured as module-level constants).

---

**14. `get_logger` ignores level changes on repeated calls**

The `if not logger.handlers` guard prevents duplicate handlers, but also prevents level updates. Calling `get_logger("foo", "DEBUG")` and then `get_logger("foo", "WARNING")` keeps DEBUG. The level should be set unconditionally:

```python
logger = logging.getLogger(name)
logger.setLevel(getattr(logging, level.upper(), logging.INFO))
if not logger.handlers:
    ...
```

---

**15. `capture.py:86-87` — misleading operator precedence**

```python
packet_count = args.packet_count or config['capture']['packet_count'] if not args.live else 0
```

Python parses this as `args.packet_count or (config[...] if not args.live else 0)`. If `args.live` is True but `args.packet_count` was explicitly provided, `packet_count` equals `args.packet_count` rather than `0`. Works in practice because argparse returns `None` for unprovided args, but is easy to misread. Use explicit parentheses or an `if/else` block.

---

**16. `random_forest.py` warm_start loop — unnecessary overhead**

Calling `model.fit(X, y)` 100 times with `warm_start=True` and incrementing `n_estimators` by 1 each time adds one tree per call. While functionally correct, it invokes 100 `fit()` call overhead cycles. The progress UX is the main benefit; a cleaner approach is a custom sklearn callback or simply `model.fit(X, y)` once with a progress estimate — or accept the loop with a comment explaining the intent.

---

**17. No `__init__.py` files in any package**

`core/`, `models/`, `utils/`, `siem/`, `cli/`, `tests/test_models/`, `tests/test_utils/` have no `__init__.py`. This works when running from the project root (`.` is on `sys.path`) but can break in edge cases — installed package mode, IDE import analysis, or test runners with non-standard working directories.

---

**18. Wazuh syslog output is not RFC 5424-compliant**

`forward_alert` sends raw JSON over UDP to the syslog port. Wazuh's network syslog input expects a formatted syslog header (priority, version, timestamp, hostname, app-name). Without it, Wazuh will likely reject or misparse the messages. For file-mode this isn't an issue — file-based Wazuh integration reads raw JSON fine. But `syslog` and `both` modes as currently implemented would not work with a standard Wazuh setup.

---

## Dependencies

**19. `requests` and `pyshark` are declared but never imported**

Neither `requests` nor `pyshark` appear in any `.py` file in the project. They add to install time and surface area with no benefit. Remove them from `pyproject.toml`, `environment.yml`, and `requirements.txt`.

---

**20. Version defined in two places**

`__version__.py` and `pyproject.toml` both define `1.0.9`. They can drift. The modern approach is to remove `__version__.py` and read it at runtime:

```python
from importlib.metadata import version
__version__ = version("anomaly-detect")
```

---

**21. `setup.py` alongside `pyproject.toml`**

Having both is legacy. `pyproject.toml` is the PEP 517/518 standard and sufficient. `setup.py` can be removed; it currently reads `__version__` from `__version__.py` and duplicates all the dependency declarations, creating a second place they can diverge.

---

## Summary

| Severity | Location | Issue |
|---|---|---|
| Bug | `tests/test_models/test_autoencoder.py:44` | `self.model_path` AttributeError — save/load test has never passed |
| Bug | `tests/test_utils/test_progress.py:6` | `TqdmKerasCallback` import fails — entire test module unloadable |
| Bug | `core/preprocessor.py:189` | `--label` writes `"anomaly"` column; supervised trainers expect `"label"` |
| Bug | `models/detector.py:51-60` | Autoencoder runs inference twice; breaks `BaseModel` abstraction |
| Bug | `utils/packet_utils.py:72` | Bare `except:` swallows `KeyboardInterrupt`, `SystemExit`, etc. |
| Bug | `core/dataset_utils.py:84` | `test_size` parameter accepted, validated, but never used |
| Bug | `core/dataset_utils.py:53,63` | Cross-file deduplication never fires due to `source` column ordering |
| Bug | `siem/wazuh_forwarder.py:34` | `forward_alert` mutates the caller's dict in-place |
| Bug | `utils/file_saver.py:127` | Phantom `placeholder.tmp` path used to create directory |
| Bug | `core/dataset_utils.py:136-142` | `balance_labels` minority/majority inverted; swap always fires |
| Design | All modules | `get_config()` at import time — breaks any non-root execution |
| Design | `models/autoencoder.py` | `AutoencoderModel` skips `super().__init__()` |
| Design | `siem/wazuh_forwarder.py` | SIEM config frozen as module-level constants at import |
| Design | `utils/logger.py` | `get_logger` silently ignores level changes on repeated calls |
| Design | `core/capture.py:86-87` | Operator precedence on ternary makes `--live` + `--packet-count` silently wrong |
| Design | `models/random_forest.py` | 100 `fit()` calls via warm_start loop — unnecessary overhead |
| Design | All packages | No `__init__.py` in any package or test subdirectory |
| Design | `siem/wazuh_forwarder.py` | Syslog output is raw JSON, not RFC 5424 — syslog/both modes non-functional with Wazuh |
| Deps | `pyproject.toml`, `environment.yml`, `requirements.txt` | `requests` and `pyshark` declared but never imported |
| Deps | `__version__.py` + `pyproject.toml` | Version string duplicated across two files |
| Deps | `setup.py` | Redundant alongside `pyproject.toml`; duplicates all dependency declarations |

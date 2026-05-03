# utils/progress.py

import sys
import threading
import time
from tqdm import tqdm


def tqdm_bar(iterable, desc="", unit="it", total=None, leave=True, disable=False):
    """
    Wraps an iterable with a tqdm progress bar.

    Parameters:
        iterable: Any iterable to wrap.
        desc (str): Description shown by progress bar.
        unit (str): Label for iteration unit (e.g., 'pkt' or 'file').
        total (int, optional): Total expected iterations.
        leave (bool): Whether to keep the bar after completion.
        disable (bool): Disables the progress bar entirely if True.

    Returns:
        tqdm: A tqdm-wrapped iterable.
    """
    return tqdm(iterable, desc=desc, unit=unit, total=total, leave=leave, disable=disable)


class TrainingSpinner:
    """
    Thread-based spinner for training operations.

    Displays a rotating braille character with live key-value stats and elapsed
    time on a single overwriting line. Call update() from model training loops
    to push per-step metrics (e.g. epoch, loss, tree count).

    Usage:
        with TrainingSpinner("Training SVM") as spinner:
            model.fit(X, y)                          # no updates - elapsed only

        with TrainingSpinner("Training RF") as spinner:
            for i in range(n):
                model.fit(X, y)
                spinner.update({"tree": f"{i+1}/{n}"})
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, desc: str = "Training"):
        self._desc = desc
        self._stats: dict = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def update(self, stats: dict) -> None:
        """Push new key-value pairs to display alongside the spinner."""
        with self._lock:
            self._stats.update(stats)

    def _spin(self) -> None:
        frame_idx = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._start_time
            elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            with self._lock:
                stats = dict(self._stats)
            parts = " | ".join(f"{k}: {v}" for k, v in stats.items())
            right = f" | {parts}" if parts else ""
            line = f"\r{self._FRAMES[frame_idx % len(self._FRAMES)]}  {self._desc}{right} | {elapsed_str}"
            sys.stderr.write(line)
            sys.stderr.flush()
            frame_idx += 1
            time.sleep(0.1)

    def start(self) -> "TrainingSpinner":
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, success: bool = True) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        mark = "✓" if success else "✗"
        # \033[K clears from the cursor to end of line, handling any length of previous content
        sys.stderr.write(f"\r{mark}  {self._desc} - done in {elapsed_str}\033[K\n")
        sys.stderr.flush()

    def __enter__(self) -> "TrainingSpinner":
        return self.start()

    def __exit__(self, exc_type, *_) -> None:
        self.stop(success=exc_type is None)

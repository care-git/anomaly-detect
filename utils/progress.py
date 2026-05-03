# utils/progress.py

import sys
import threading
import time
from tqdm import tqdm


def _ansi_supported() -> bool:
    """
    Returns True if the current terminal supports ANSI escape sequences.

    On Windows, attempts to enable Virtual Terminal Processing via the Win32
    console API. Legacy CMD and older PowerShell hosts that cannot enable it
    will receive False, triggering the ASCII fallback in TrainingSpinner.
    All UNIX-like terminals return True unconditionally.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import ctypes.wintypes
        STD_ERROR_HANDLE = -12
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_ERROR_HANDLE)
        mode = ctypes.wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False


_ANSI = _ansi_supported()


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

    Displays a rotating character with live key-value stats and elapsed time on
    a single overwriting line. Uses braille frames and ANSI escape sequences on
    terminals that support them; falls back to ASCII frames and space-padding on
    Windows legacy terminals (CMD, older PowerShell) that do not.

    Call update() from model training loops to push per-step metrics (e.g.
    epoch, loss, tree count).

    Usage:
        with TrainingSpinner("Training SVM") as spinner:
            model.fit(X, y)                          # no updates - elapsed only

        with TrainingSpinner("Training RF") as spinner:
            for i in range(n):
                model.fit(X, y)
                spinner.update({"tree": f"{i+1}/{n}"})
    """

    _FRAMES_ANSI  = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _FRAMES_ASCII = r"|/-\|/-\ "
    _MARKS_ANSI   = ("✓", "✗")
    _MARKS_ASCII  = ("done", "FAILED")

    def __init__(self, desc: str = "Training"):
        self._desc = desc
        self._stats: dict = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._frames = self._FRAMES_ANSI if _ANSI else self._FRAMES_ASCII
        self._marks  = self._MARKS_ANSI  if _ANSI else self._MARKS_ASCII

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
            line = f"\r{self._frames[frame_idx % len(self._frames)]}  {self._desc}{right} | {elapsed_str}"
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
        mark = self._marks[0] if success else self._marks[1]
        if _ANSI:
            # \033[K clears from cursor to end of line, handling any length of previous content
            sys.stderr.write(f"\r{mark}  {self._desc} - done in {elapsed_str}\033[K\n")
        else:
            line = f"\r{mark}  {self._desc} - done in {elapsed_str}"
            sys.stderr.write(line.ljust(79) + "\n")
        sys.stderr.flush()

    def __enter__(self) -> "TrainingSpinner":
        return self.start()

    def __exit__(self, exc_type, *_) -> None:
        self.stop(success=exc_type is None)

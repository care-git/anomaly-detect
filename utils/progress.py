# utils/progress.py

from contextlib import contextmanager
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

@contextmanager
def single_bar(desc="", total=1, unit="it", leave=True, disable=False):
    """
    Context manager for single-step progress bars.

    Useful for operations I can't reasonably assign an iterable for.

    Parameters:
        desc (str): Description shown by progress bar.
        total (int): Total count (default is 1).
        unit (str): Label for iteration unit.
        leave (bool): Whether to keep the bar after completion.
        disable (bool): Disables the progress bar entirely if True.
    """
    bar = tqdm(total=total, desc=desc, unit=unit, leave=leave, disable=disable)
    yield lambda: bar.update(1)
    bar.close()

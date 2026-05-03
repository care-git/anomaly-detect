from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("anomaly-detect")
except PackageNotFoundError:
    # Running from source without installation - use the file written by setuptools-scm.
    try:
        from _version import __version__
    except ImportError:
        __version__ = "unknown"

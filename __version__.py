from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("anomaly-detect")
except PackageNotFoundError:
    __version__ = "unknown"

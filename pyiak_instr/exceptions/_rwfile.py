from pathlib import Path

from ._base import PyiError


__all__ = [
    "RWFileError",
    "FileSuffixError"
]


class RWFileError(PyiError):
    """
    Base class for exceptionts in rwfile module
    """

    def __init__(self, msg: str, filepath: Path):
        super().__init__(msg, filepath)
        self.filepath = filepath


class FileSuffixError(RWFileError):
    """
    Raised when filepath is wrong by some pattern
    """

    def __init__(self, suffixes: set[str], filepath: Path):
        super().__init__(
            "suffix of '%s' not in %s" % (filepath, suffixes),
            filepath
        )

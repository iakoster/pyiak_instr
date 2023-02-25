from pathlib import Path
from typing import Any

from ._base import PyiError


__all__ = ["RWFileError", "FileSuffixError"]


class RWFileError(PyiError):
    """Base class for exceptions in rwfile package."""

    def __init__(self, msg: str, filepath: Path, *args: Any):
        super().__init__(msg, filepath, *args)
        self.filepath = filepath


class FileSuffixError(RWFileError):
    """Raised when filepath is wrong by some pattern."""

    def __init__(self, suffixes: set[str], filepath: Path):
        super().__init__(
            "suffix of '%s' not in %s" % (filepath, suffixes),
            filepath,
            filepath.suffix,
        )
        self.suffixes = suffixes

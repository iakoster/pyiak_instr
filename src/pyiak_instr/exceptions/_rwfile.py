"""
Private module of `pyiak_instr.exceptions` with exceptions for rwfile
package.
"""
from pathlib import Path
from typing import Any

from ._base import PyiError


__all__ = ["RWFileError", "FileSuffixError"]


class RWFileError(PyiError):
    """
    Base class for exceptions in rwfile package.

    Parameters
    ----------
    msg: str
        exception message.
    filepath: Path
        path to the file.
    *args: Any
        exception arguments.
    """

    def __init__(self, msg: str, filepath: Path, *args: Any):
        super().__init__(filepath, *args, msg=msg)
        self.filepath = filepath


class FileSuffixError(RWFileError):
    """
    Raised when filepath has invalid suffix.

    Parameters
    ----------
    suffixes: set[str]
        allowed suffixes.
    filepath: Path
        path to the file with invalid suffix.
    """

    def __init__(self, suffixes: set[str], filepath: Path):
        super().__init__(
            f"suffix of '{filepath}' not in {suffixes}",
            filepath,
            filepath.suffix,
        )
        self.suffixes = suffixes

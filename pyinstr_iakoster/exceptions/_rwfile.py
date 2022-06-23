from pathlib import Path

from ._base import PyiError


__all__ = [
    "FilepathPatternError"
]


class FilepathPatternError(PyiError):
    """
    Raised when filepath is wrong by some pattern
    """

    def __init__(self, pattern: str, filepath: Path | str):
        PyiError.__init__(
            self, 'The path does not lead to %r file' % pattern)
        if isinstance(filepath, str):
            filepath = Path(filepath)
        self.filepath = filepath
        self.pattern = pattern
        self.args = (pattern, filepath)

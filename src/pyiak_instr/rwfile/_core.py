"""Private module of ``pyiak_instr.osutils._win`` with common functions for
Windows.
"""
from __future__ import annotations
from pathlib import Path
from abc import abstractmethod
from types import TracebackType
from typing import Any, Optional, Type, TypeVar


from ..exceptions import RWFileError, FileSuffixError
from ..typing import ContextManager, WithApi, WithBaseStringMethods


__all__ = [
    "RWFile",
    "RWFileError",
    "FileSuffixError",
]


T = TypeVar("T")


def _convert_path(filepath: Path | str) -> Path:
    """
    Convert `filepath` to Path if `filepath` is str instance.

    Parameters
    ----------
    filepath : Path or path-like string
        path to file

    Returns
    -------
    Path
        path as Path instance.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    return filepath


def _check_filepath(cls: type[RWFile[T]], filepath: Path | str) -> None:
    """
    Check that `filepath` is correct.

    Parameters
    ----------
    cls: RWFile
        class instance.
    filepath : Path
        path to the file.

    Raises
    ------
    FileSuffixError
        if suffix of `filepath` is not in `ALLOWED_SUFFIXES`.
    FileNotFoundError
        if `filepath` exists, and it is not a file.
    """
    filepath_ = _convert_path(filepath)
    if (
        len(cls.ALLOWED_SUFFIXES)
        and filepath_.suffix not in cls.ALLOWED_SUFFIXES
    ):
        raise FileSuffixError(cls.ALLOWED_SUFFIXES, filepath_)

    if filepath_.exists() and not filepath_.is_file():
        raise FileNotFoundError("path not lead to file")
    filepath_.absolute().parent.mkdir(parents=True, exist_ok=True)


class RWFile(ContextManager, WithApi[T], WithBaseStringMethods):
    """
    Represents a base class for read/write file.

    Parameters
    ----------
    filepath: Path | str
        path to the file.
    api: T
        api for work with file.
    """

    ALLOWED_SUFFIXES: set[str] = set()
    "Allowed file suffixes."

    def __init__(self, filepath: Path | str, api: T):
        WithApi.__init__(self, api)
        self._fp = _convert_path(filepath)

    @abstractmethod
    def close(self) -> None:
        """Close api."""

    def _get_under_brackets(self) -> str:
        """
        Returns
        -------
        str
            string which must be under brackets in result of .__str__ method.
        """
        return str(self._fp)

    @property
    def filepath(self) -> Path:
        """
        Returns
        -------
        Path
            Path to the file.
        """
        return self._fp

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
        return super().__exit__(exc_type, exc_val, exc_tb)

    # pylint: disable=unused-argument
    def __new__(
        cls, filepath: Path | str, *args: Any, **kwargs: Any
    ) -> RWFile[T]:
        _check_filepath(cls, filepath)
        return super().__new__(cls)

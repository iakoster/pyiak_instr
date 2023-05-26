"""Private module of ``pyiak_instr.osutils._win`` with common functions for
Windows.
"""
from __future__ import annotations
from pathlib import Path
from abc import abstractmethod
from types import TracebackType
from typing import Any, Optional, Self, Type, TypeVar


from ...exceptions import FileSuffixError, NotSupportedMethod
from ...typing import ContextManager, WithApi, WithBaseStringMethods


__all__ = ["RWData"]


ApiT = TypeVar("ApiT")


def check_filepath(cls: type[RWData[ApiT]], filepath: Path) -> None:
    """
    Check that `filepath` is correct.

    Parameters
    ----------
    cls: RWData
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
    if (
        len(cls.ALLOWED_SUFFIXES)
        and filepath.suffix not in cls.ALLOWED_SUFFIXES
    ):
        raise FileSuffixError(cls.ALLOWED_SUFFIXES, filepath)

    if filepath.exists() and not filepath.is_file():
        raise FileNotFoundError("path not lead to file")
    filepath.absolute().parent.mkdir(parents=True, exist_ok=True)


# todo: expand to any data
# todo: split to RWData and RWFile
class RWData(ContextManager, WithApi[ApiT], WithBaseStringMethods):
    """
    Represents a base class for read/write file.

    Parameters
    ----------
    filepath: Path
        path to the file.
    """

    ALLOWED_SUFFIXES: set[str] = set()
    "Allowed file suffixes."

    def __init__(self, filepath: Path):
        WithApi.__init__(self, self._get_api(filepath))
        self._fp = filepath

    @abstractmethod
    def commit(self) -> None:
        """Apply/save changes."""

    @abstractmethod
    def close(self) -> None:
        """Close api."""

    @abstractmethod
    def _get_api(self, filepath: Path) -> ApiT:
        """
        Get api instance.

        Returns
        -------
        ApiT
        """

    def request(self, *args: Any, **kwargs: Any) -> Any:
        """
        Send some request.

        Parameters
        ----------
        *args : Any
            positional function arguments.
        **kwargs : Any
            keyword arguments.

        Returns
        -------
        Any
            method result.

        Raises
        ------
        NotSupportedMethod
            if method is declared, but not implemented.
        """
        raise NotSupportedMethod(self.request.__qualname__)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        """
        Get data from memory.

        Parameters
        ----------
        *args : Any
            positional function arguments.
        **kwargs : Any
            keyword arguments.

        Returns
        -------
        Any
            method result.

        Raises
        ------
        NotSupportedMethod
            if method is declared, but not implemented.
        """
        raise NotSupportedMethod(self.get.__qualname__)

    def set(self, *args: Any, **kwargs: Any) -> Any:
        """
        Set data to memory.

        Parameters
        ----------
        *args : Any
            positional function arguments.
        **kwargs : Any
            keyword arguments.

        Returns
        -------
        Any
            method result.

        Raises
        ------
        NotSupportedMethod
            if method is declared, but not implemented.
        """
        raise NotSupportedMethod(self.set.__qualname__)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        """
        Read data from local disk.

        Parameters
        ----------
        *args : Any
            positional function arguments.
        **kwargs : Any
            keyword arguments.

        Returns
        -------
        Any
            method result.

        Raises
        ------
        NotSupportedMethod
            if method is declared, but not implemented.
        """
        raise NotSupportedMethod(self.read.__qualname__)

    def write(self, *args: Any, **kwargs: Any) -> Any:
        """
        Write data to local disk.

        Parameters
        ----------
        *args : Any
            positional function arguments.
        **kwargs : Any
            keyword arguments.

        Returns
        -------
        Any
            method result.

        Raises
        ------
        NotSupportedMethod
            if method is declared, but not implemented.
        """
        raise NotSupportedMethod(self.write.__qualname__)

    @property
    def filepath(self) -> Path:
        """
        Returns
        -------
        Path
            Path to the file.
        """
        return self._fp

    def __str_under_brackets__(self) -> str:
        """
        Returns
        -------
        str
            string which must be under brackets in result of .__str__ method.
        """
        return f"'{self._fp}'"

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __new__(
        cls, filepath: Path, *args: Any, **kwargs: Any
    ):  # todo: typing - Self[T]
        check_filepath(cls, filepath)
        return super().__new__(cls)

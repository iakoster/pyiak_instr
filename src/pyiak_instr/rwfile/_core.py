from pathlib import Path
from abc import abstractmethod
from types import TracebackType
from typing import Optional, Type, TypeVar


from ..exceptions import RWFileError, FileSuffixError
from ..typing import ContextManager, WithApi, WithBaseStringMethods


__all__ = [
    "RWFile",
    "RWFileError",
    "FileSuffixError",
]


T = TypeVar("T")


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

        if isinstance(filepath, str):
            filepath = Path(filepath)
        if (
            len(self.ALLOWED_SUFFIXES)
            and filepath.suffix not in self.ALLOWED_SUFFIXES
        ):
            raise FileSuffixError(self.ALLOWED_SUFFIXES, filepath)

        if filepath.exists() and not filepath.is_file():
            raise FileNotFoundError("path not lead to file")
        filepath.absolute().parent.mkdir(parents=True, exist_ok=True)

        self._fp = filepath

    @abstractmethod
    def close(self) -> None:
        """Close api."""
        pass

    def _get_under_brackets(self) -> str:
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

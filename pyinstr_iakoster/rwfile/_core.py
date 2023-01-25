from pathlib import Path
from typing import Generic


from ..core import UseApi, API_TYPE
from ..exceptions import FileSuffixError, RWFileError


__all__ = [
    "RWFile",
    "RWFileError",
    "FileSuffixError",
]


class RWFile(UseApi[API_TYPE], Generic[API_TYPE]):
    """
    Represents a base class for read/write file.

    Parameters
    ----------
    filepath: Path or str
        path to the file.
    """

    FILE_SUFFIXES: set[str] = {}
    "Allowed file suffixes."

    def __init__(self, filepath: Path | str):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if len(self.FILE_SUFFIXES) and \
                filepath.suffix not in self.FILE_SUFFIXES:
            raise FileSuffixError(self.FILE_SUFFIXES, filepath)

        if filepath.exists():
            if not filepath.is_file():
                raise RWFileError("path not lead to file", filepath)
        elif not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

        self._fp = filepath

    def close(self) -> None:
        """
        Close connection to the file.
        """
        raise NotImplementedError()

    @property
    def filepath(self) -> Path:
        """
        Returns
        -------
        Path
            Path to the file.
        """
        return self._fp

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            String interpretation of the class instance.
        """
        return f"<{str(self)}>"

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            String interpretation of the class instance.
        """
        return f"{self.__class__.__name__}('{self._fp}')"


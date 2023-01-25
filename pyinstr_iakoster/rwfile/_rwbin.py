from pathlib import Path

from ._core import RWFile

__all__ = ["RWBin"]


class RWBin(RWFile[None]):

    FILE_SUFFIXES = {".bin"}

    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._api = None

    def close(self) -> None:
        pass

    def read_all(self) -> bytes:
        """
        Read all content from file.

        Returns
        -------
        bytes
            file content.
        """
        with open(self._fp, "rb") as file:
            return file.read()

    def rewrite(self, content: bytes) -> None:
        """
        Rewrite file content.

        Parameters
        ----------
        content: bytes
            new content of the file.
        """
        with open(self._fp, "wb") as file:
            file.write(content)

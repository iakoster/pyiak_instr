import deprecation
from pathlib import Path

from tinydb import TinyDB
from tinydb.table import Table, Document

from ._core import RWFile

__all__ = [
    "RWNSDocument",
    "RWNoSqlJsonDatabase",
]


RWNSDocument = Document


@deprecation.deprecated(deprecated_in="0.0.1.1", removed_in="0.0.2.0")
class RWNoSqlJsonDatabase(RWFile):
    """
    The class of the modified TinyDB.

    Aimed to work only with JSON files.

    See Also
    --------
    tinydb.TinyDB: parent class.

    """

    FILE_SUFFIXES = {".json"}

    def __init__(self, filepath: Path | str):
        super().__init__(filepath)
        self._hapi = TinyDB(filepath)

    def close(self) -> None:
        self._hapi.close()

    @property
    def hapi(self):
        return self._hapi

    def __getitem__(self, table: str) -> Table:
        """
        Returns
        -------
        Table
            database table.
        """
        return self._hapi.table(table)

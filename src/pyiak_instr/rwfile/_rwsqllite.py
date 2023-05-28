"""Private module of ``pyiak_instr.rwfile`` with class for work with
SQLite3 database.
"""
import sqlite3
from pathlib import Path
from typing import Any

from .types import RWData


__all__ = ["RWSQLite"]


class RWSQLite(RWData[sqlite3.Cursor]):
    """
    Class for reading and writing to the database as *.db.

    The timeout specifies how long the connection should
    wait for the lock to go away until raising an exceptions.

    Parameters
    ----------
    filepath: Path
        path to the database.
    autocommit: bool, default=True
        commit after any changes.
    timeout: int, default=5
        database response timeout.
    """

    ALLOWED_SUFFIXES = {".db"}
    """Set of the allowed suffixes"""

    def __init__(
        self,
        filepath: Path,
        autocommit: bool = True,
        timeout: float = 5,
    ):
        self._con = sqlite3.connect(filepath, timeout=timeout)
        super().__init__(filepath)
        self._autocommit = autocommit

    def request(
        self,
        request: str,
        many: list[tuple[Any, ...]] | tuple[Any, ...] | None = None,
    ) -> sqlite3.Cursor:
        """
        Execute command and return resulting cursor.

        Parameters
        ----------
        request: str
            sql command.
        many: list[tuple[Any, ...]] | tuple[Any, ...] | None, default=None
            execute parameters or list of parameters.

        Returns
        -------
        sqlite3.Cursor
            sql cursor.
        """
        if many is None:
            result = self._api.execute(request)
        elif isinstance(many, tuple):
            result = self._api.execute(request, many)
        else:
            result = self._api.executemany(request, many)

        if self._autocommit:
            self.commit()
        return result

    def close(self) -> None:
        """Close connection to database."""
        try:
            self._api.close()
        except sqlite3.ProgrammingError as err:
            if err.args[0] != "Cannot operate on a closed database.":
                raise
        try:
            self._con.close()
        except sqlite3.ProgrammingError as err:
            if err.args[0] != "Cannot operate on a closed database.":
                raise

    def commit(self) -> None:
        """Commit changes."""
        self._con.commit()

    def _get_api(self, filepath: Path) -> sqlite3.Cursor:
        """
        Get cursor of database connection.

        Parameters
        ----------
        filepath :
            ignored. Needed for backward compatibility.

        Returns
        -------
        sqlite3.Cursor
            connection cursor.
        """
        # pylint: disable=unused-argument
        return self._con.cursor()

    @property
    def tables(self) -> list[str]:
        """
        List of table names in the database.

        The table names are taken from the 'sqlite_master' table.

        Returns
        -------
        list of str
            table names in the database.
        """
        return [
            el[0]
            for el in self.request(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

    @property
    def columns(self) -> dict[str, list[str]]:
        """
        The column names in the tables.

        Presented in format {table: [column, ...]}.

        Returns
        -------
        dict[str, list[str]]
            list of columns in all tables.
        """
        return {
            t: [e[0] for e in self.request(f"SELECT * FROM {t};").description]
            for t in self.tables
        }

    @property
    def rows_counts(self) -> dict[str, int]:
        """
        The count of rows in each table in the database.

        Returns
        -------
        dict[str, int]
            row count in each table in format {table: rows count}.
        """

        return {
            table: self.request(f"SELECT COUNT(*) FROM {table};").fetchone()[
                0
            ]
            for table in self.tables
        }

    @property
    def connection(self) -> sqlite3.Connection:
        """
        Returns
        -------
        sqlite3.Connection
            SQLite connection to the database.
        """
        return self._con

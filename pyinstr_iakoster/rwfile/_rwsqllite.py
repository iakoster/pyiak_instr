import sqlite3
from pathlib import Path

from ._core import RWFile


__all__ = ['RWSQLite']


class RWSQLite(RWFile[sqlite3.Cursor]):
    """
    Class for reading and writing to the database as *.db.

    The timeout specifies how long the connection should
    wait for the lock to go away until raising an exception.

    Parameters
    ----------
    filepath: Path or path-like str
        path to the database.
    autocommit: bool, default=True
        commit after any changes.
    timeout: int, default=5
        database response timeout.
    """

    FILE_SUFFIXES = {".db"}

    def __init__(
            self,
            filepath: Path | str,
            autocommit: bool = True,
            timeout: float = 5
    ):
        super().__init__(filepath)

        self._conn = sqlite3.connect(filepath, timeout=timeout)
        self._api = self._conn.cursor()
        self._autocommit = autocommit

    def request(
            self,
            request: str,
            many: list[tuple] | tuple = None
    ) -> sqlite3.Cursor:
        """
        Execute command and return resulting cursor.

        Parameters
        ----------
        request: str
            sql command.
        many: list of tuples or tuple
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

    def create_table(self, table: str, **col_pars: str) -> None:
        """
        Execute sql command
        'CREATE TABLE IF NOT EXISTS {table} ({col_pars});'.

        Column parameters forming from the dict as 'key value',
        where key is a column name and value is column parameters
        (e.g. 'TEXT' or 'INT PRIMARY KEY').

        Parameters
        ----------
        table: str
            table name.
        **col_pars: str
            column parameters in format {column: parameters}
        """

        self.request('CREATE TABLE IF NOT EXISTS {}({});'.format(
            table, ', '.join(f'{k} {v}' for k, v in col_pars.items())))

        if self._autocommit:
            self.commit()

    def table_columns(self, table: str) -> list[str]:
        """
        Get column names in the table.

        Parameters
        ----------
        table: str
            table name.

        Returns
        -------
        list of str
            list of a column names in the table
        """
        self.request('SELECT * FROM %s;' % table)
        return [el[0] for el in self._api.description]

    def table_rows(self, table: str) -> int:
        """
        Get rows count in the table.

        Parameters
        ----------
        table: str
            table name.

        Returns
        -------
        int
            rows count in the table.
        """
        return self.request(
            'SELECT COUNT(*) FROM %s;' % table
        ).fetchone()[0]

    def commit(self) -> None:
        """Commit changes."""
        self._conn.commit()

    def close(self) -> None:
        """Close cursor and connection."""
        try:
            self._api.close()
        except sqlite3.ProgrammingError as err:
            if err.args[0] != 'Cannot operate on a closed database.':
                raise
        try:
            self._conn.close()
        except sqlite3.ProgrammingError as err:
            if err.args[0] != 'Cannot operate on a closed database.':
                raise

    @property
    def tables(self) -> list[str]:
        """
        List of table names in the database.

        The table names are taken from
        the 'sqlite_master' table.

        Returns
        -------
        list of str
            table names in the database.
        """
        return [el[0] for el in self.request(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

    @property
    def columns(self) -> dict[str, list[str]]:
        """
        The column names in the tables.
        Presented in format {table: [column, ...]}.

        Returns
        -------
        dict of {str: list of str}
            list of columns in all tables.
        """
        return {table: self.table_columns(table)
                for table in self.tables}

    @property
    def rows(self) -> dict[str, int]:
        """
        The count of rows in each table in the database.

        Returns
        -------
        dict of {str: int}
            row count in each table in format {table: rows count}.
        """
        return {table: self.table_rows(table)
                for table in self.tables}

    @property
    def connection(self) -> sqlite3.Connection:
        """
        Returns
        -------
        sqlite3.Connection
            SQLite connection to the database.
        """
        return self._conn

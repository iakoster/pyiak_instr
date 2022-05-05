import re
import sqlite3
from pathlib import Path
from typing import Iterable

import deprecation
import pandas as pd

from ._utils import *


__all__ = ['RWSQLite3Simple']


class RWSQLite3Simple(object):
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

    FILENAME_PATTERN = re.compile('\S+.db$')

    def __init__(
            self,
            filepath: Path | str,
            autocommit: bool = True,
            timeout: float = 5
    ):
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath
        self._conn = sqlite3.connect(filepath, timeout=timeout)
        self._cur = self._conn.cursor()
        self._autocommit = autocommit

    def request(
            self, request: str,
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
            result = self._cur.execute(request)
        elif isinstance(many, tuple):
            result = self._cur.execute(request, many)
        else:
            result = self._cur.executemany(request, many)

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

    def insert_into(
            self, *,
            insert_into: str,
            values: tuple | list[tuple],
            columns: list[str] = None
    ) -> None:
        """
        Execute sql command
        'INSERT INTO {TABLE} VALUES(?, ?, ...);'.
        Can insert a list of rows.

        The values parameter must be represented as
        tuple or list of tuples, where each tuple must
        have the same len as a selected row.

        Parameters
        ----------
        insert_into: str
            where to insert.
        values: tuple or list of tuple
            cell values.
        columns: list of str, default=None
            selected columns to insert.
        """
        if isinstance(values, tuple):
            val_marks = ', '.join(['?'] * len(values))
        else:
            val_marks = ', '.join(['?'] * len(values[0]))
        columns = '' if columns is None else \
            '({})'.format(', '.join(columns))

        self.request('INSERT INTO {}{} VALUES ({});'.format(
            insert_into, columns, val_marks), values)

    @deprecation.deprecated(
        deprecated_in='0.0.4', removed_in='0.0.6',
        details='it is redundant function')
    def delete_from(self, *, from_: str) -> None:
        """
        Execute sql command 'DELETE FROM {table};'.

        Parameters
        ----------
        from_: str
            from where to delete.
        """
        request = 'DELETE FROM %s;' % from_
        self._cur.execute(request)

        if self._autocommit:
            self.commit()

    @deprecation.deprecated(
        deprecated_in='0.0.4', removed_in='0.0.6',
        details='it is redundant function')
    def select(
            self, *, from_: str, select: str = '*',
            fetch: int | str = None, where: str = None):
        """
        Execute sql command 'SELECT {select} FROM {from_};'.

        Append 'WHERE {where}' to the request if
        where parameter is not None.

        On the result can be applyed method fetchmany
        if fetch is integer and fetch > 0 or
        fetchall if fetch == 'all'.

        Parameters
        ----------
        from_: str
            from where to be select.
        select: str, default='*'
            what to be select.
        fetch: int or str, optional
            fetch result and how or not.
        where: str, optional
            where statement in the request.

        Returns
        -------
        sqlite3.Cursor or list of Any
            result of an operation.

        Raises
        ------
        ValueError
            if fetch is instance of int and less than one.
        ValueError
            if fetch is not instance of int, not None or
            not equal 'all'.
        """

        where = '' if where is None else f' WHERE {where}'
        request = 'SELECT {} FROM {}{};'.format(select, from_, where)
        result = self._cur.execute(request)

        if fetch is None:
            return result
        elif isinstance(fetch, int):
            if fetch < 1:
                raise ValueError('fetch cannot be less than 1')
            return result.fetchmany(fetch)
        elif fetch == 'all':
            return result.fetchall()
        else:
            raise ValueError('unknown fetch variable %r' % fetch)

    @deprecation.deprecated(
        deprecated_in='0.0.4', removed_in='0.0.6',
        details='it is redundant function')
    def to_dataframe(
            self, *, from_: str, select: str = '*',
            where: str = None, index_col=None
    ) -> pd.DataFrame:
        """
        Execute sql command 'SELECT {select} FROM {from_};'.

        Append 'WHERE {where}' to the request if
        where parameter is not None.

        Parameters
        ----------
        from_: str
            from where to be select.
        select: str, default='*'
            what to be select.
        where: str, optional
            where statement in the request
        index_col: str or list of str, optional
            column(s) to set as index(MultiIndex).

        Returns
        -------
        pd.DataFrame
            sql table as a pandas DataFrame.
        """

        where = '' if where is None else f' WHERE {where}'
        request = 'SELECT {} FROM {}{};'.format(select, from_, where)
        return pd.read_sql_query(
            request, self._conn, index_col=index_col)

    @deprecation.deprecated(
        deprecated_in='0.0.4', removed_in='0.0.6',
        details='it is redundant function')
    def from_dataframe(
            self, *,
            df: pd.DataFrame, table: str,
            if_exists: str = 'replace',
            index: bool = True,
            index_label=None
    ) -> None:
        """
        Export pandas dataframe to sql database.

        Parameters
        ----------
        df: pd.DataFrame,
            dataframe.
        table: str
            table name.
        if_exists: str, default='replace'
            How to behave if the table already exists.
        index: bool, default=True
            Write DataFrame index as a column.
            Uses `index_label` as the column name in the table.
        index_label: str or sequence, default=None
            Column label for index column(s).
            If None is given (default) and `index` is True,
            then the index names are used.
        """

        df.to_sql(
            table, self._conn, if_exists=if_exists,
            index=index, index_label=index_label)

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
        return [el[0] for el in self._cur.description]

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
            self._cur.close()
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
        The column names in the each table.
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
    def connection(self):
        """
        Returns
        -------
        sqlite3.Connection
            SQLite connection to the database.
        """
        return self._conn

    @property
    def cursor(self):
        """
        Returns
        -------
        sqlite3.Cursor
            cursor of the SQLite connection.
        """
        return self._cur

    @property
    def filepath(self):
        """
        Returns
        -------
        Path
            path to the SQL database
        """
        return self._filepath

    def __del__(self):
        self.close()

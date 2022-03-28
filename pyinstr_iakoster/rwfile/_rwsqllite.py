import re
import sqlite3
from pathlib import Path

import pandas as pd

from ._rwf_utils import *


class RWSQLite3Simple(object):
    """
    Class for reading and writing to the database as *.db.
    """

    FILENAME_PATTERN = re.compile('\w+.db$')

    def __init__(
            self,
            filepath: Path | str,
            autocommit: bool = True,
            timeout: float = 5
    ):
        """
        :param filepath: path to the database
        :param autocommit: commit after any changes
        """
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath
        self._conn = sqlite3.connect(filepath, timeout=timeout)
        self._cur = self._conn.cursor()
        self._autocommit = autocommit

    def create_table(self, table: str, **col_pars: str) -> None:
        """
        execute sql command 'CREATE TABLE IF NOT EXISTS {table}
        ({col_pars});'.

        Column parameters forming from the dict as 'key value',
        where key is a column name and value is column parameters
        (e.g. 'TEXT' or 'INT PRIMARY KEY')

        :param table: table name
        :param col_pars: column parameters
        """
        request = 'CREATE TABLE IF NOT EXISTS {}({});'.format(
            table,
            ', '.join(f'{col} {par}' for col, par in
                      col_pars.items()))
        self._cur.execute(request)

        if self._autocommit:
            self.commit()

    def insert_into(
            self, *,
            into: str,
            values: tuple | list[tuple],
            columns: list[str] = None
    ) -> None:
        """
        execute sql command 'INSERT INTO {table} VALUES({values});'.
        Can insert a list of rows

        :param into: where to insert
        :param values: tuple of a cell values
        :param columns: list of columns for insert values
        """
        if isinstance(values, tuple):
            values = [values]
        columns = '' if columns is None else \
            '({})'.format(', '.join(columns))
        val_marks = ', '.join(['?'] * len(values[0]))

        request = 'INSERT INTO {}{} VALUES ({});'.format(
            into, columns, val_marks)
        self._cur.executemany(request, values)

        if self._autocommit:
            self.commit()

    def delete_from(self, *, from_: str) -> None:
        """
        execute sql command 'DELETE FROM {table};

        :param from_: from where to delete
        """
        request = 'DELETE FROM %s;' % from_
        self._cur.execute(request)

        if self._autocommit:
            self.commit()

    def select(
            self, *, from_: str, select: str = '*',
            fetch: int | str = None, where: str = None):
        """
        execute sql command 'SELECT {select} FROM {from_}'

        Append 'WHERE {where}' to the request if where is not None

        On the result can be applyed method
        fetchmany if fetch is integer and fetch > 0 or
        fetchall if fetch == 'all'.

        :param select: what to be select
        :param from_: from where to be select
        :param fetch: fetch result and how or not
        :param where: where statement in the request
        :return: result of an operation
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

    def to_dataframe(
            self, *, from_: str, select: str = '*',
            where: str = None, index_col=None
    ) -> pd.DataFrame:
        """
        execute sql command 'SELECT {select} FROM {from_}'

        Append 'WHERE {where}' to the request if where is not None

        :param select: what to be select
        :param from_: from where to be select
        :param where: where statement in the request
        :param index_col: column(s) to set as index(MultiIndex).
        :return: pandas dataframe
        """
        where = '' if where is None else f' WHERE {where}'
        request = 'SELECT {} FROM {}{};'.format(select, from_, where)
        return pd.read_sql_query(
            request, self._conn, index_col=index_col)

    def from_dataframe(
            self, *,
            df: pd.DataFrame, table: str,
            if_exists: str = 'replace',
            index: bool = True,
            index_label=None
    ) -> None:
        """
        export pandas dataframe to sql database

        :param df: dataframe
        :param table: table name
        :param index: Write DataFrame index as a column.
            Uses `index_label` as the column name in the table.
        :param index_label: Column label for index column(s).
            If None is given (default) and `index` is True,
            then the index names are used.
        :param if_exists: How to behave if the table already exists.
        """
        df.to_sql(
            table, self._conn, if_exists=if_exists,
            index=index, index_label=index_label)

    def table_columns(self, table: str) -> list[str]:
        """
        :param table: table name
        :return: list of a column names in the table
        """
        self._cur.execute('SELECT * FROM %s;' % table)
        return list(map(lambda x: x[0], self._cur.description))

    def table_rows_count(self, table: str) -> int:
        """
        :param table: table name
        :return: rows count in the table
        """
        return self._cur.execute('SELECT COUNT(*) FROM %s;' % table).fetchall()[0][0]

    def commit(self) -> None:
        """
        commit changes
        """
        self._conn.commit()

    def close(self) -> None:
        """
        close cursor and connection
        """
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
        :return: list of tables in the database
        """
        return list(map(lambda el: el[0], self.select(
            select='name', from_='sqlite_master',
            where='type=\'table\'', fetch='all')))

    @property
    def columns(self) -> dict[str, list[str]]:
        """
        :return: list of columns in all tables
        """
        columns = {}
        for table in self.tables:
            columns[table] = self.table_columns(table)
        return columns

    @property
    def rows_count(self) -> dict[str, int]:
        rows_count = {}
        for table in self.tables:
            rows_count[table] = self.table_rows_count(table)
        return rows_count

    @property
    def connection(self):
        """
        :return: sqllite connection to the database
        """
        return self._conn

    @property
    def cursor(self):
        """
        :return: cursor of the connection
        """
        return self._cur

    @property
    def filepath(self):
        """
        :return: path to the sql database
        """
        return self._filepath

    def __del__(self):
        self.close()

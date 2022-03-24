import re
import sqlite3
from pathlib import Path

from ._rwf_utils import *


class RWSimpleSqlLite(object):
    """
    Class for reading and writing to the database as *.db.
    """

    FILENAME_PATTERN = re.compile('\w+.db$')

    def __init__(self, filepath: Path | str, autocommit: bool = True):
        """
        :param filepath: path to the database
        :param autocommit: commit after any changes
        """
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath
        self._conn = sqlite3.connect(filepath)
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
        request = 'CREATE TABLE IF NOT EXISTS {} ({});'.format(
            table,
            ', '.join(f'{col} {par}' for col, par in
                      col_pars.items()))
        self._cur.execute(request)

        if self._autocommit:
            self.commit()

    def insert_into(self, into: str, values: tuple | list[tuple]) -> None:
        """
        execute sql command 'INSERT INTO {table} VALUES({values});'.
        Can insert a list of rows

        :param into: table name
        :param values: tuple of cell values
        """
        if isinstance(values, list):
            cols_count = len(values[0])
            operation = self._cur.executemany
        else:
            cols_count = len(values)
            operation = self._cur.execute

        request = 'INSERT INTO {} VALUES({});'.format(
            into, ', '.join(['?'] * cols_count))
        operation(request, values)

        if self._autocommit:
            self.commit()

    def select(self, select: str = '*', from_: str = '*', fetch: int | str = None):
        """
        execute sql command 'SELECT {select} FROM {from_}'

        On the result can be applyed method
        fetchmany if fetch is integer and fetch > 0 or
        fetchall if fetch == 'all'.

        :param select: what will be selected
        :param from_: from where will be selected
        :param fetch: fetch result and how or not
        :return: result of a operation
        """
        request = 'SELECT {} FROM {};'.format(select, from_)
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
            raise ValueError('unknown fetch variabe %r' % fetch)

    def commit(self) -> None:
        """
        commit changes
        """
        self._conn.commit()

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
        self._cur.close()
        self._conn.close()

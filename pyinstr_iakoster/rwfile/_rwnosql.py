import re
from pathlib import Path
from typing import Any, Iterable

from tinydb import TinyDB
from tinydb.table import Table, QueryLike

from ._utils import if_str2path, create_dir_if_not_exists, match_filename


__all__ = [
    "RWNoSqlJsonDatabase",
    "RWNoSqlTable",
]


class RWNoSqlTable(Table):
    """
    Represents a single modified TinyDB table.

    See Also
    --------
    tinydb.table.Table: parent class.

    """

    def rwf_insert(self, **fields: Any) -> int:
        """
        Insert a new document into the table.

        Parameters
        ----------
        **fields: Any
            parameters for new document.

        Returns
        -------
        int
            the inserted document's ID.
        """
        return Table.insert(self, fields)

    def rwf_update(
            self,
            cond: QueryLike = None,
            doc_ids: Iterable[int] | int = None,
            **fields: Any
    ) -> list[int]:
        """
        Update all matching documents to have a given set of fields.

        Parameters
        ----------
        cond: QueryLike
            which documents to update.
        doc_ids: iterable if ints or int
            document ID('s) to update.
        **fields: Any
            the fields that the matching documents will have or a method
            that will update the documents.

        Returns
        -------
        list of ints
            the updated document's ID.
        """
        if isinstance(doc_ids, int):
            doc_ids = (doc_ids,)
        return Table.update(self, fields, cond=cond, doc_ids=doc_ids)

    def rwf_upsert(self, cond: QueryLike = None, **fields) -> list[int]:
        """
        Update documents, if they exist, insert them otherwise.

        Parameters
        ----------
        cond: QueryLike
            which document to look for.
        **fields: Any
            the document parameters to insert or the fields to update.

        Returns
        -------
        list of ints
            the updated documents' IDs.

        See Also
        --------
        tinydb.table.Table.upsert: original method.

        """
        return Table.upsert(self, fields, cond=cond)


class RWNoSqlJsonDatabase(TinyDB):
    """
    The class of the modified TinyDB.

    Aimed to work only with JSON files.

    See Also
    --------
    tinydb.TinyDB: parent class.

    """

    table_class = RWNoSqlTable
    _tables: dict[str, table_class]
    FILENAME_PATTERN = re.compile('\S+.json$')

    def __init__(self, filepath: Path | str, *args, **kwargs):
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)
        self._filepath = filepath

        super().__init__(filepath, *args, **kwargs)

    rwf_insert = RWNoSqlTable.rwf_insert
    rwf_update = RWNoSqlTable.rwf_update
    rwf_upsert = RWNoSqlTable.rwf_upsert

    def table(self, name: str, **kwargs: Any) -> table_class:
        """
        Get access to a RWNoSqlTable.

        Parameters
        ----------
        name: str
            The name of the table.
        **kwargs: Any
            Keyword arguments to pass to the table class constructor.

        Returns
        -------

        See Also
        --------
        tinydb.TinyDB: original method.

        """
        if name in self._tables:
            return self._tables[name]
        table = self.table_class(self.storage, name, **kwargs)
        self._tables[name] = table
        return table

    @property
    def filepath(self):
        """
        Returns
        -------
        Path
            path to the JSON database/file.
        """
        return self._filepath

    def __getitem__(self, table: str) -> table_class:
        """
        Get access to a RWNoSqlTable.

        See Also
        --------
        self.table: this magic uses the specified method.

        """
        return self.table(table)

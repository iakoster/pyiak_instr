import re
from pathlib import Path
from typing import Any, Iterable

from tinydb import TinyDB, Query
from tinydb.table import (
    Table, Document, QueryLike
)

from ._utils import *


__all__ = [
    "RWNoSqlJsonDatabase", "RWNoSqlTable",
    "Query", "Document"
]


class RWNoSqlTable(Table):

    def rwf_insert(self, **doc_fields: Any) -> int:
        return Table.insert(self, doc_fields)

    def rwf_update(
            self,
            cond: QueryLike = None,
            doc_ids: Iterable[int] | int = None,
            **fields: Any
    ) -> list[int]:
        if isinstance(doc_ids, int):
            doc_ids = (doc_ids,)
        return Table.update(self, fields, cond=cond, doc_ids=doc_ids)

    def rwf_upsert(self, cond: QueryLike = None, **fields) -> list[int]:
        return Table.upsert(self, fields, cond=cond)


class RWNoSqlJsonDatabase(TinyDB):

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
        if name in self._tables:
            return self._tables[name]
        table = self.table_class(self.storage, name, **kwargs)
        self._tables[name] = table
        return table

    @property
    def filepath(self):
        return self._filepath

    def __getitem__(self, table: str) -> table_class:
        return self.table(table)

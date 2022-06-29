import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._mess import FieldSetter, Message
from ..rwfile import (
    RWSQLite3Simple,
    match_filename,
    create_dir_if_not_exists
)
from ..utilities import StringEncoder


__all__ = [
    "PackageFormat"
]


class PackageFormatBase(object):

    FILENAME_PATTERN = re.compile("\S+.db$")

    def __init__(
            self,
            msg_settings: dict[str, Any],
            **setters: FieldSetter
    ):
        self._msg_sets = msg_settings
        self._setters = setters

    @property
    def msg_settings(self):
        return self._msg_sets

    @property
    def setters(self) -> dict[str, FieldSetter]:
        return self._setters


class PackageFormat(PackageFormatBase):

    def write_pf(self, path: Path):
        match_filename(self.FILENAME_PATTERN, path)
        create_dir_if_not_exists(path)

        msg_sets = pd.DataFrame(
            columns=["name", "value"], data=((
                k, StringEncoder.to_str(v)
            ) for k, v in self._msg_sets.items())
        )
        setters = pd.DataFrame(
            columns=["name", "special", "kwargs"],
            data=((
                n, s.special, StringEncoder.to_str(s.kwargs)
            ) for n, s in self._setters.items())
        )
        fmt_name = self._msg_sets["format_name"]

        with sqlite3.connect(path) as con:
            msg_sets.to_sql(
                f"{fmt_name}__msg_sets",
                con,
                if_exists="replace",
                index=False
            )
            setters.to_sql(
                f"{fmt_name}__setters",
                con,
                if_exists="replace",
                index=False
            )

    @classmethod
    def read_pf(cls, path: Path, fmt_name: str):

        if not path.exists():
            raise FileExistsError("file %s not exists" % path)
        match_filename(cls.FILENAME_PATTERN, path)
        with sqlite3.connect(path) as con:
            df_msg_sets = pd.read_sql(f"SELECT * FROM {fmt_name}__msg_sets;", con)
            df_setters = pd.read_sql(f"SELECT * FROM {fmt_name}__setters;", con)

        setters = {}
        for i in range(len(df_setters)):
            row = df_setters.iloc[i]
            setters[row["name"]] = FieldSetter(
                special=row["special"], **StringEncoder.from_str(row["kwargs"])
            )
        msg_sets = {}
        for i in range(len(df_msg_sets)):
            row = df_msg_sets.iloc[i]
            msg_sets[row["name"]] = StringEncoder.from_str(row["value"])

        return cls(msg_sets, **setters)



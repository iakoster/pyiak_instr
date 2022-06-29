import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy.types as sqlt

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

    SQL_TYPES = {
        "format_name": "TEXT",
        "splittable": "BOOL",
        "slice_length": "INT",
        "name": "TEXT",
        "special": "TEXT",
        "fmt": "TEXT",
        "content": "UNSIGNED INT",
        "may_be_empty": "BOOL",
        "units": "UNSIGNED INT",
        "additive": "UNSIGNED INT",
        "desc_dict": "TEXT",
        "expected": "INT",
    }

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

        fmt_name = self._msg_sets["format_name"]
        df_msg_sets = pd.DataFrame(
            columns=self._msg_sets.keys(), data=[self._msg_sets.values()]
        )
        df_setters = pd.DataFrame()
        for i_set, (name, setter) in enumerate(self._setters.items()):
            df_setters.loc[i_set, "name"] = name
            df_setters.loc[i_set, "special"] = setter.special
            for par, val in setter.kwargs.items():
                if isinstance(val, dict):
                    val = StringEncoder.to_str(val)
                if val is not None:
                    df_setters.loc[i_set, par] = val

        with sqlite3.connect(path) as con:
            df_msg_sets.to_sql(
                f"{fmt_name}__msg_sets",
                con,
                if_exists="replace",
                index=False,
            )
            df_setters.to_sql(
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

        msg_sets = {}
        for col in df_msg_sets:
            msg_sets[col] = df_msg_sets[col].iloc[0]

        setters = {}
        for i_row in range(len(df_setters)):
            row: pd.Series = df_setters.iloc[i_row]
            name, special = row["name"], row["special"]
            kwargs_ser = row.drop(["name", "special"]).dropna()
            kwargs = {}
            for col in kwargs_ser.index:
                val = kwargs_ser[col]
                if isinstance(val, str):
                    val = StringEncoder.from_str(val)
                kwargs[col] = val
            setters[name] = FieldSetter(special=special, **kwargs)

        return cls(msg_sets, **setters)



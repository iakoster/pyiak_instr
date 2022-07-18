import re
import sqlite3
import inspect
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy.types as sqlt

from ._msg import FieldSetter, Message
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
    SETTER_CLASS = FieldSetter
    MESSAGE_CLASS = Message

    def __init__(
            self,
            **settings: FieldSetter | Any
    ):
        self._settings = {}
        self._setters = {}
        for k, v in settings.items():
            if isinstance(v, FieldSetter):
                self._setters[k] = v
            else:
                self._settings[k] = v

        setters_diff = set(self._setters) - set(Message.REQ_FIELDS)
        if len(setters_diff):
            ValueError(
                f"not all requared setters were got: %s are missing" %
                ", ".join(setters_diff)
            )

    @property
    def msg_args(self) -> list[str]:
        return list(
            inspect.getfullargspec(
                self.MESSAGE_CLASS.__init__
            ).annotations.keys()
        )

    @property
    def settings(self) -> dict[str, Any]:
        return self._settings

    @property
    def setters(self) -> dict[str, FieldSetter]:
        return self._setters

    @property
    def fields_args(self) -> list[str]:
        args = ["name", "special"]
        for name, method in inspect.getmembers(
                self.SETTER_CLASS(), predicate=inspect.ismethod
        ):
            if "of <class 'pyinstr" not in repr(method):
                continue

            for par in inspect.getfullargspec(method).annotations.keys():
                if par not in args:
                    args.append(par)
        return args


class PackageFormat(PackageFormatBase):

    def write_pf(self, path: Path):
        match_filename(self.FILENAME_PATTERN, path)
        create_dir_if_not_exists(path)
        fmt_name = self._settings["format_name"]

        df_setters = self._get_empty_dataframe()
        for i_set, (name, setter) in enumerate(self._setters.items()):
            for par, val in itertools.chain(
                [("name", name), ("special", setter.special)],
                setter.kwargs.items()
            ):
                if isinstance(val, dict):
                    val = StringEncoder.to_str(val)
                if val is not None:
                    df_setters.loc[i_set, par] = val

        with sqlite3.connect(path) as con:
            pd.DataFrame(
                columns=self.msg_args, data=[self._settings.values()]
            ).to_sql(
                f"{fmt_name}__settings",
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

    def _get_empty_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.fields_args)

    @classmethod
    def read_pf(cls, path: Path, fmt_name: str):

        if not path.exists():
            raise FileExistsError("file %s not exists" % path)
        match_filename(cls.FILENAME_PATTERN, path)
        with sqlite3.connect(path) as con:
            msg_sets = pd.read_sql(
                f"SELECT * FROM {fmt_name}__settings;", con
            ).iloc[0].to_dict()
            df_setters = pd.read_sql(
                f"SELECT * FROM {fmt_name}__setters;", con
            )

        setters = {}
        for i_row in range(len(df_setters)):
            row: pd.Series = df_setters.iloc[i_row]
            name, special = row["name"], row["special"]
            row = row.drop(["name", "special"]).dropna()

            str_mask = (row.apply(type) == str)
            row[str_mask] = row[str_mask].apply(StringEncoder.from_str)
            setters[name] = FieldSetter(special=special, **row.to_dict())

        return cls(**msg_sets, **setters)



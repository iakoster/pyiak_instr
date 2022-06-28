import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._mess import FieldSetter, Message
from ..rwfile import (
    match_filename,
    create_dir_if_not_exists
)
from ..utilities import StringConverter


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

        msg_sets = pd.DataFrame(columns=["name", "value"])
        for name, value in self._msg_sets.items():
            msg_sets.loc[msg_sets.shape[0]] = [
                name, StringConverter.to_str(value)
            ]

        setters = pd.DataFrame(columns=["name", "special", "kwargs"])
        for name, setter in self._setters.items():
            setters.loc[setters.shape[0]] = [
                name, setter.special, StringConverter.to_str(setter.kwargs)
            ]

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
    def read_pf(cls, path: Path):

        if not path.exists():
            raise FileExistsError("file %s not exists" % path)
        match_filename(cls.FILENAME_PATTERN, path)


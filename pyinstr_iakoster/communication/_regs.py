from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, field, InitVar

import sqlite3
from pathlib import Path

import pandas as pd

from ..rwfile import RWSQLite3Simple

if TYPE_CHECKING:
    from ._msg import Message, ContentType
    from ._pf import MessageFormat, PackageFormat


__all__ = [
    "Register",
    "RegisterMap"
]


@dataclass(frozen=True, eq=False)
class Register(object): # nodesc

    extended_name: str
    name: str
    format_name: str
    address: int
    length: int
    reg_type: str = "rw"
    data_fmt: str = None
    description: str = ""
    mf: MessageFormat = None

    def __post_init__(self):
        if self.reg_type not in {"rw", "ro", "wo"}:
            raise TypeError("invalid register type: %r" % self.reg_type)

    def read(
            self,
            data_length: ContentType = None,
            update: dict[str, dict[str, Any]] = None,
            **other_fields: ContentType
    ) -> Message: # nodesc
        if self.reg_type == "wo":
            raise TypeError("writing only") # todo unique exception

        msg = self._get_message(**self._modify_update_kw(update or {}))
        return self._validate_msg(msg.set(
            address=self.address,
            operation=other_fields["operation"]
            if "operation" in other_fields else
            self._find_operation(msg, "r"),
            data_length=data_length or self.length,
            **other_fields
        ))

    def write(
            self,
            data: ContentType = b"",
            update: dict[str, dict[str, Any]] = None,
            **other_fields: ContentType
    ) -> Message: # nodesc
        if self.reg_type == "ro":
            raise TypeError("reading only") # todo unique exception

        msg = self._get_message(**self._modify_update_kw(update or {}))
        return self._validate_msg(msg.set(
            address=self.address,
            operation=other_fields["operation"]
            if "operation" in other_fields else
            self._find_operation(msg, "w"),
            data=data,
            **other_fields
        ))

    def _get_message(self, **update: dict[str, Any]) -> Message:
        if self.mf is None:
            raise AttributeError("message format not specified")  # todo: custom exception
        return self.mf.get(**update)

    def _modify_update_kw(
            self, update: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        if self.data_fmt is not None and (
            "data" not in update or
            "data" in update and "fmt" not in update["data"]
        ):
            if "data" in update:
                tmp = update["data"]
                tmp["fmt"] = self.data_fmt
                update.update(data=tmp)
            else:
                update["data"] = {"fmt": self.data_fmt}
        return update

    def _validate_msg(self, msg: Message) -> Message: # nodesc
        dlen = msg.data_length.unpack()[0]
        if dlen > self.length:
            raise ValueError(
                "invalid data length: %d > %d" % (dlen, self.length)
            )

        return msg

    @classmethod
    def from_series(
            cls, series: pd.Series, mf: MessageFormat = None
    ) -> Register: # nodesc
        return cls(**series.to_dict(), mf=mf)

    @staticmethod
    def _find_operation(msg: Message, base: str) -> str: # nodesc
        assert len(base) == 1
        for msg_oper in msg.operation.desc_dict:
            if msg_oper[0] == base:
                return msg_oper
        raise ValueError("operation starts with %r not found" % base)

    @property
    def short_description(self) -> str: # nodesc
        short = ""
        for letter in self.description:
            short += letter
            if letter == ".":
                break
        return short # todo: may use itertools


class RegisterMap(object): # nodesc

    EXPECTED_COLUMNS = (
        "extended_name",
        "name",
        "format_name",
        "address",
        "reg_type",
        "length",
        "data_fmt",
        "description",
    )

    def __init__(
            self,
            registers_table: pd.DataFrame
    ):
        self._tbl = self._validate_table(registers_table)

    def get(self, name: str, pf: PackageFormat = None) -> Register: # nodesc
        name_table = self._tbl[self._tbl["name"] == name]
        ext_table = self._tbl[self._tbl["extended_name"] == name]

        if len(name_table):
            assert len(name_table) == 1
            series = name_table.iloc[0]
        elif len(ext_table):
            assert len(ext_table) == 1
            series = ext_table.iloc[0]
        else:
            raise AttributeError("register %r not found" % name)

        return Register.from_series(
            series, mf=None if pf is None else pf[series["format_name"]]
        )

    def write(self, con: sqlite3.Connection) -> None: # nodesc
        self._tbl.to_sql("registers", con, index=False)

    def _validate_table(self, table: pd.DataFrame) -> pd.DataFrame: # nodesc
        cols_diff = set(table.columns) - set(self.EXPECTED_COLUMNS)
        if len(cols_diff):
            raise ValueError(f"invalid columns: {cols_diff}") # todo: custom exc
        # todo: checks:
        #  without dublicates in extended_name and name
        #  without duplicates in address with the same msg_fmt
        #  sort by msg_fmt and address
        return table.sort_values(by=["format_name", "address"])

    @classmethod
    def read(cls, database: Path) -> RegisterMap: # nodesc
        with RWSQLite3Simple(database, autocommit=False) as db:
            return RegisterMap(
                pd.read_sql("SELECT * FROM registers", db.connection)
            )

    @property
    def table(self) -> pd.DataFrame: # nodesc
        return self._tbl

    def __getattr__(self, name: str) -> Register: # nodesc
        return self.get(name)

    def __getitem__(self, name: str | tuple[str, PackageFormat]) -> Register: # nodesc
        if isinstance(name, tuple):
            name, pf = name
        else:
            pf = None
        return self.get(name, pf=pf)



from __future__ import annotations
from typing import TYPE_CHECKING, Any

import sqlite3
from pathlib import Path

import pandas as pd

from ..rwfile import RWSQLite3Simple

if TYPE_CHECKING:
    from ._msg import Message, ContentType
    from ._pf import MessageFormat


__all__ = [
    "Register",
    "RegisterMap"
]


class Register(object): # nodesc

    RO = 0  # todo: change to enum codes
    WO = 1
    RW = 2
    REG_TYPES = {"RO": RO, "WO": WO, "RW": RW}
    _REG_TYPES_R = {v: k for k, v in REG_TYPES.items()}

    _mf: MessageFormat

    def __init__(
            self,
            extended_name: str,
            name: str,
            format_name: str,
            address: int,
            reg_type: str | int,
            length: int,
            data_fmt: str = None,
            description: str = ""
    ):
        if isinstance(reg_type, int) and reg_type in self._REG_TYPES_R:
            pass
        elif isinstance(reg_type, str) and reg_type.upper() in self.REG_TYPES:
            reg_type = self.REG_TYPES[reg_type.upper()]
        else:
            raise TypeError("invalid register type: %r" % reg_type)

        self._ext_name = extended_name
        self._name = name
        self._fmt_name = format_name
        self._addr = address
        self._type = reg_type
        self._len = length
        self._dfmt = data_fmt
        self._desc = description

    def read(
            self,
            data_length: ContentType = None,
            update: dict[str, dict[str, Any]] = None,
            **other_fields: ContentType
    ) -> Message: # nodesc
        if self._type == self.WO:
            raise TypeError("writing only") # todo unique exception

        msg = self._get_message(**self._modify_update_kw(update or {}))
        return self._validate_msg(msg.set(
            address=self._addr,
            operation=other_fields["operation"]
            if "operation" in other_fields else
            self._find_operation(msg, "r"),
            data_length=data_length or self._len,
            **other_fields
        ))

    def set_message_format(self, mf: MessageFormat) -> Register: # nodesc
        self._mf = mf
        return self

    def write(
            self,
            data: ContentType = b"",
            update: dict[str, dict[str, Any]] = None,
            **other_fields: ContentType
    ) -> Message: # nodesc
        if self._type == self.RO:
            raise TypeError("reading only") # todo unique exception

        msg = self._get_message(**self._modify_update_kw(update or {}))
        return self._validate_msg(msg.set(
            address=self._addr,
            operation=other_fields["operation"]
            if "operation" in other_fields else
            self._find_operation(msg, "w"),
            data=data,
            **other_fields
        ))

    def _get_message(self, **update: dict[str, Any]) -> Message:
        if not hasattr(self, "_mf"):
            raise AttributeError("message format not specified")  # todo: custom exception
        msg = self._mf.get(**update)
        del self._mf
        return msg

    def _modify_update_kw(
            self, update: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        if self._dfmt is not None and (
            "data" not in update or
            "data" in update and "fmt" not in update["data"]
        ):
            if "data" in update:
                tmp = update["data"]
                tmp["fmt"] = self._dfmt
                update.update(data=tmp)
            else:
                update["data"] = {"fmt": self._dfmt}
        return update

    def _validate_msg(self, msg: Message) -> Message: # nodesc
        dlen = msg.data_length.unpack()[0]
        if dlen > self._len:
            raise ValueError(
                "invalid data length: %d > %d" % (dlen, self._len)
            )

        return msg

    @classmethod
    def from_series(cls, series: pd.Series) -> Register: # nodesc
        return cls(**series.to_dict())

    @staticmethod
    def _find_operation(msg: Message, base: str) -> str: # nodesc
        assert len(base) == 1
        for msg_oper in msg.operation.desc_dict:
            if msg_oper[0] == base:
                return msg_oper
        raise ValueError("operation in message not founded")

    @property
    def address(self) -> int: # nodesc
        return self._addr

    @property
    def data_fmt(self) -> str: # nodesc
        return self._dfmt

    @property
    def description(self) -> str: # nodesc
        return self._desc

    @property
    def extended_name(self) -> str: # nodesc
        return self._ext_name

    @property
    def format_name(self) -> str: # nodesc
        return self._fmt_name

    @property
    def length(self) -> int: # nodesc
        return self._len

    @property
    def name(self) -> str: # nodesc
        return self._name

    @property
    def reg_type(self) -> int: # nodesc
        return self._type

    @property
    def reg_type_str(self) -> str: # nodesc
        return self._REG_TYPES_R[self._type]

    @property
    def short_description(self) -> str: # nodesc
        short = ""
        for letter in self._desc:
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

    def get(self, name: str) -> Register: # nodesc
        name_table = self._tbl[self._tbl["name"] == name]
        if len(name_table):
            assert len(name_table) == 1
            return Register.from_series(name_table.iloc[0])

        ext_table = self._tbl[self._tbl["extended_name"] == name]
        if len(ext_table):
            assert len(ext_table) == 1
            return Register.from_series(ext_table.iloc[0])

        raise AttributeError("register %r not found" % name)

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

    def __getitem__(self, name: str) -> Register: # nodesc
        return self.get(name)



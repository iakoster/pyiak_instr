from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._msg import Message, ContentType
    from ._pf import MessageFormat


__all__ = [
    "Register",
    "RegisterMap"
]


class Register(object): # nodesc

    _mf: MessageFormat

    def __init__(
            self,
            extended_name: str,
            name: str,
            address: int,
            length: int,
            message_format_name: str,
            description: str = ""
    ):
        self._ext_name = extended_name
        self._name = name
        self._addr = address
        self._dlen = length
        self._msg_fmt_name = message_format_name
        self._desc = description

    def read(
            self,
            data_length: ContentType = None,
            **other_fields: ContentType
    ) -> Message: # nodesc
        if "_mf" not in dir(self):
            raise AttributeError("message format not setted") # todo: custom exception
        msg = self._mf.get()
        del self._mf
        return msg.set(
            address=self._addr,
            operation=self._find_operation(msg, "r"),
            data_length=self._dlen if data_length is None else data_length,
            **other_fields
        )

    def set_message_format(self, mf: MessageFormat) -> Register: # nodesc
        self._mf = mf
        return self

    def write(
            self,
            data: ContentType,
            **other_fields: ContentType
    ) -> Message: # nodesc
        if "_mf" not in dir(self):
            raise AttributeError("message format not setted")  # todo: custom exception
        msg = self._mf.get()
        del self._mf
        return msg.set(
            address=self._addr,
            operation=self._find_operation(msg, "w"),
            data=data,
            **other_fields
        )

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
    def description(self) -> str: # nodesc
        return self._desc

    @property
    def extended_name(self) -> str: # nodesc
        return self._ext_name

    @property
    def length(self) -> int: # nodesc
        return self._dlen

    @property
    def message_format_name(self) -> str: # nodesc
        return self._msg_fmt_name

    @property
    def name(self) -> str: # nodesc
        return self._name

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
        "address",
        "length",
        "message_format_name",
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

        raise AttributeError("register %r not exists" % name)

    def _validate_table(self, table: pd.DataFrame) -> pd.DataFrame: # nodesc
        cols_diff = set(table.columns) - set(self.EXPECTED_COLUMNS)
        if len(cols_diff):
            raise ValueError(f"invalid columns: {cols_diff}") # todo: custom exc
        # todo: checks:
        #  without dublicates in extended_name and name
        #  without duplicates in address with the same msg_fmt
        #  sort by msg_fmt and address
        return table.sort_values(by=["message_format_name", "address"])

    @property
    def table(self) -> pd.DataFrame: # nodesc
        return self._tbl

    def __getattr__(self, name: str) -> Register: # nodesc
        return self.get(name)

    def __getitem__(self, name: str) -> Register: # nodesc
        return self.get(name)



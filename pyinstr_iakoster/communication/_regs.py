from __future__ import annotations

from itertools import takewhile
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

import sqlite3
from pathlib import Path

import pandas as pd

from ..rwfile import RWSQLite

if TYPE_CHECKING:
    from ._msg import Message, ContentType
    from ._pf import MessageFormat, PackageFormat


__all__ = [
    "Register",
    "RegisterMap"
]


@dataclass(frozen=True, eq=False)
class Register(object):
    """
    Represents class instance of register.

    Raises
    ------
    TypeError
        if `reg_type` not in {'rw', 'ro', 'wo'}.
    """

    external_name: str | None
    "name of the register in external documents."

    name: str
    "the name of the register"

    format_name: str
    "the name of the message format."

    address: int
    "register address. Used for address field in message."

    length: int
    "register length. May used for `data_length` field in message."

    reg_type: str = "rw"
    "register type. Can be one of {'rw', 'ro', 'wo'}"

    data_fmt: str | None = None
    "format of a data in the register."

    description: str = ""
    "register description. First sentence must be a short summary."

    mf: MessageFormat | None = None
    "message format for messages of this register."

    def __post_init__(self):
        if self.reg_type not in {"rw", "ro", "wo"}:
            raise TypeError("invalid register type: %r" % self.reg_type)

    def shift(self, shift: int) -> Register:
        """
        Shift the address by a specified number.

        Shift value must be in range [0,length). Also adds the suffix
        '_shifted' to the name (if the name does not end in '_shifted').

        If shift value is equal to zero returns the same register
        instance and create new register otherwise.

        Parameters
        ----------
        shift: int
            address offset.

        Returns
        -------
        Register
            shifted register instance.

        Raises
        ------
        ValueError
            if shift value is negative;
            if shift more or equal than register length.
        """

        if shift == 0:
            return self
        elif shift < 0:
            raise ValueError("shift can't be negative")
        elif shift >= self.length:
            raise ValueError("shift more or equal to register length")
        elif self.name.endswith("_shifted"):
            name = self.name
        else:
            name = self.name + "_shifted"

        return Register(
            self.external_name,
            name,
            self.format_name,
            self.address + shift,
            self.length - shift,
            reg_type=self.reg_type,
            data_fmt=self.data_fmt,
            description=self.description,
            mf=self.mf
        )

    def read(
            self,
            data_length: ContentType = None,
            update: dict[str, dict[str, Any]] = None,
            **other_fields: ContentType
    ) -> Message:
        """
        Get message with read operation.

        If 'operation' not specified in `other_fields`, find operation
        in `desc_dict` by operation base.

        Parameters
        ----------
        data_length: ContentType, default=None
            The length of the data for reading. If None than length will be
            equal length of the register.
        update: dict[str, dict[str, Any]], default=None
            parameters for update standard settings from MessageFormat.
            Must be writter in format {FIELD_NAME: {FIELD_ATTR: VALUE}}.
        **other_fields: ContentType, optional
            values for .set method of message.

        Returns
        -------
        Message
            message for reading data from register.

        Raises
        ------
        TypeError
            if register is write only.
        """
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
    ) -> Message:
        """
        Get message with write operation.

        If 'operation' not specified in `other_fields`, find operation
        in `desc_dict` by operation base.

        Parameters
        ----------
        data: ContentType, default=b''
            The length of the data for writing.
        update: dict[str, dict[str, Any]], default=None
            parameters for update standard settings from MessageFormat.
            Must be writter in format {FIELD_NAME: {FIELD_ATTR: VALUE}}.
        **other_fields: ContentType, optional
            values for .set method of message.

        Returns
        -------
        Message
            message for writing data from register.

        Raises
        ------
        TypeError
            if register is read only.
        """
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
        """
        Get message from message format.

        Parameters
        ----------
        **update: dict[str, Any], optional
            parameters for update standard settings from MessageFormat.

        Returns
        -------
        Message
            message with specified message format.

        Raises
        ------
        AttributeError
            if message format is not specified.
        """
        if self.mf is None:
            raise AttributeError("message format not specified")  # todo: custom exception
        return self.mf.get(**update)

    def _modify_update_kw(
            self, update: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Modify update kwargs.

        If {'data': {'fmt': VALUE}} not exists, set `data_fmt` from register
        it is specified.

        Parameters
        ----------
        update: dict[str, dict[str, Any]]
            parameters for update standard settings from MessageFormat.

        Returns
        -------
        dict[str, dict[str, Any]]
            modified update kwargs.
        """
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

    def _validate_msg(self, msg: Message) -> Message:
        """
        Check message to correct settings.

        Parameters
        ----------
        msg: Message
            message.

        Returns
        -------
        Message
            message instance.

        Raises
        ------
        ValueError
            if `data_length` in message more than register length.
        """
        dlen = msg.data_length.unpack()[0]
        if dlen > self.length:
            raise ValueError(
                "invalid data length: %d > %d" % (dlen, self.length)
            )

        return msg

    @classmethod
    def from_series(
            cls, series: pd.Series, mf: MessageFormat = None
    ) -> Register:
        """
        Get register fron pandas.Series

        Parameters
        ----------
        series: pandas.Series
            series with register parameters.
        mf: MessageFormat, default=None.
            message format for this register.

        Returns
        -------
        Register
            register instance.
        """
        return cls(**series.to_dict(), mf=mf)

    @staticmethod
    def _find_operation(msg: Message, base: str) -> str:
        """
        Find operation from `desc_dict` from `operation` field by base.

        Parameters
        ----------
        msg: Message
            message.
        base: str
            operation base.

        Returns
        -------
        str
            full operation name from `desc_dict`.

        Raises
        ------
        ValueError
            if there is no operations starts with `base`.
        """
        assert len(base) == 1
        for msg_oper in msg.operation.desc_dict:
            if msg_oper[0] == base:
                return msg_oper
        raise ValueError("operation starts with %r not found" % base)

    @property
    def short_description(self) -> str:
        """
        Returns first sentence from description.

        Returns
        -------
        str
            Short description.
        """
        return "".join(takewhile(lambda l: l != ".", self.description))

    @property
    def series(self) -> pd.Series:
        """
        Returns
        -------
        pandas.Series
            register parameters in series.
        """
        return pd.Series(
            index=RegisterMap.EXPECTED_COLUMNS,
            data=(
                self.external_name,
                self.name,
                self.format_name,
                self.address,
                self.length,
                self.reg_type,
                self.data_fmt,
                self.description
            )
        )

    def __add__(self, shift: int) -> Register:
        """Shift the address by a specified number."""
        return self.shift(shift)


class RegisterMap(object):
    """
    Represents class instance to store registers.

    Parameters
    ----------
    registers_table: pandas.DataFrame
        table with parameters for registers.
    """

    EXPECTED_COLUMNS = (
        "external_name",
        "name",
        "format_name",
        "address",
        "reg_type",
        "length",
        "data_fmt",
        "description",
    )
    "tuple of expected columns in `register_table`"

    def __init__(
            self,
            registers_table: pd.DataFrame
    ):
        self._tbl = self._validate_table(registers_table)

    def get(self, name: str, pf: PackageFormat = None) -> Register:
        """
        Get register by name.

        Searches the register first by 'name', then, if not found,
        by 'external_name'.

        Set to register message format if it exists in package format.

        Parameters
        ----------
        name: str
            register name.
        pf: PackageFormat
            package format which contain required message format.

        Returns
        -------
        Register
            register instance.
        """
        name_table = self._tbl[self._tbl["name"] == name]
        ext_table = self._tbl[self._tbl["external_name"] == name]

        if len(name_table):
            assert len(name_table) == 1
            series = name_table.iloc[0]
        elif len(ext_table):
            assert len(ext_table) == 1
            series = ext_table.iloc[0]
        else:
            raise AttributeError("register %r not found" % name)

        return Register.from_series(
            series, mf=None if pf is None else pf.get_format(series["format_name"])
        )

    def write(self, con: sqlite3.Connection) -> None:
        """
        Write register map table to sqlite table.

        Parameters
        ----------
        con: sqlite3.Connection
            connection to database.
        """
        self._tbl.to_sql("registers", con, index=False)

    def _validate_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Check table and data in table.

        Parameters
        ----------
        table: pandas.DataFrame
            table for validating.

        Returns
        -------
        pandas.DataFrame
            table for validating.

        Raises
        ------
        ValueError
            if not all expected columns specified
        """
        cols_diff = set(table.columns) - set(self.EXPECTED_COLUMNS)
        if len(cols_diff):
            raise ValueError(f"invalid columns: {cols_diff}") # todo: custom exc
        # todo: checks:
        #  without dublicates in external_name and name
        #  without duplicates in address with the same msg_fmt
        #  sort by msg_fmt and address
        return table.sort_values(by=["format_name", "address"])

    @classmethod
    def read(cls, database: Path) -> RegisterMap:
        """
        Read RegisterMap from database.

        Data will be readed from 'registers' table.

        Parameters
        ----------
        database: Path
            path to the database.

        Returns
        -------
        RegisterMap
            register map instance.
        """
        with RWSQLite(database, autocommit=False) as db:
            return RegisterMap(
                pd.read_sql("SELECT * FROM registers", db.connection)
            )

    @property
    def table(self) -> pd.DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            table with parameters of registers.
        """
        return self._tbl

    def __getattr__(self, name: str) -> Register:
        """
        Get register by name.

        Parameters
        ----------
        name: str
            register name.

        Returns
        -------
        Register
            register instance.
        """
        return self.get(name)

    def __getitem__(self, name: str | tuple[str, PackageFormat]) -> Register:
        """
        Get register by name.

        Parameters
        ----------
        name: str or tuple of str and PackageFormat
            register name or register name and package format.

        Returns
        -------
        Register
            register instance.
        """
        if isinstance(name, tuple):
            name, pf = name
        else:
            pf = None
        return self.get(name, pf=pf)



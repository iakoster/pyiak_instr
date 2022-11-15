from __future__ import annotations

from itertools import takewhile
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    overload,
)
from dataclasses import dataclass

import sqlite3
from pathlib import Path

import pandas as pd

from ...rwfile import RWSQLite
from ...utilities import split_complex_dict

if TYPE_CHECKING:
    from .field import ContentType
    from .message import Message
    from .package_format import MessageFormat, PackageFormat


__all__ = [
    "Register",
    "RegisterMap"
]

# todo: think through a scheme of interaction


def _validate_register_rw_input(invalid_type: str):
    """
    validate input of .read and .write methods.

    Parameters
    ----------
    invalid_type: str
        type of register when exception will be risen.

    Returns
    -------
    Callable
        function wrapper.
    """

    if invalid_type not in {"ro", "wo"}:
        raise ValueError(
            "invalid argument: %s not in {'ro', 'wo'}" % invalid_type
        )

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(
                self: Register, *args: ContentType, **kwargs: Any
        ) -> Message:
            if self.register_type == invalid_type:
                if invalid_type == "ro":
                    raise TypeError("read only register")
                raise TypeError("write only register")
            if "address" in kwargs:
                raise ValueError("setting the 'address' is not allowed")

            if len(args) and func.__name__ == "write":
                if len(args) > 1:
                    raise ValueError("to many arguments")
                kwargs["data"] = args[0]

            return func(self, **kwargs)

        return wrapper

    return decorator


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

    register_type: str = "rw"
    "register type. Can be one of {'rw', 'ro', 'wo'}"

    data__fmt: str = None
    "format of a data in the register."

    description: str = ""
    "register description. First sentence must be a short summary."

    mf: MessageFormat | None = None
    "message format for messages of this register."

    def __post_init__(self):
        if self.register_type not in {"rw", "ro", "wo"}:
            raise TypeError("invalid register type: %r" % self.register_type)

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
        elif not (0 < shift < self.length):
            raise ValueError(
                "invalid shift: %d not in [0, %d)" % (shift, self.length)
            )
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
            register_type=self.register_type,
            data__fmt=self.data__fmt,
            description=self.description,
            mf=self.mf
        )

    @_validate_register_rw_input("wo")
    def read(self, **update: Any) -> Message:
        """
        Get message with read operation.

        Update must contain two dicts: kwargs for .configure method and
        kwargs for .set method. Kwargs for .configure method must be written
        in format FIELD__PARAMETER=VALUE, where '__' is a separator between
        field name and field parameter. The other keys will be defined as
        kwargs for the .set method

        If 'operation' not specified in `update`, find operation
        in `desc_dict` by operation base.

        Parameters
        ----------
        **update: Any, optional
            parameters for update standard settings from MessageFormat.

        Returns
        -------
        Message
            message for reading data from register.

        Raises
        ------
        TypeError
            if register is write only.
        ValueError
            if 'address' in `update`.

        See Also
        --------
        pyinstr_iakoster.utilities.split_complex_dict: function for splitting
            incoming dict.
        """
        return self._get_message(
            "r", *split_complex_dict(update, without_sep="other")
        )

    @overload
    def write(self, data: ContentType, **update: ContentType) -> Message:
        ...

    @_validate_register_rw_input("ro")
    def write(self, **update: ContentType) -> Message:
        """
        Get message with write operation.

        Update must contain two dicts: kwargs for .configure method and
        kwargs for .set method. Kwargs for .configure method must be written
        in format FIELD__PARAMETER=VALUE, where '__' is a separator between
        field name and field parameter. The other keys will be defined as
        kwargs for the .set method

        If 'operation' not specified in `update`, find operation
        in `desc_dict` by operation base.

        Parameters
        ----------
        **update: Any, optional
            parameters for update standard settings from MessageFormat
            and message content.

        Returns
        -------
        Message
            message for writing data from register.

        Raises
        ------
        TypeError
            if register is read only.
        ValueError
            if 'address' in `update`.
        """
        return self._get_message(
            "w", *split_complex_dict(update, without_sep="other")
        )

    def _get_message(
            self,
            operation_base: str,
            configure_kw: dict[str, dict[str, Any]],
            set_kw: dict[str, Any]
    ) -> Message:
        """
        Get message from message format.

        Parameters
        ----------
        operation_base: str
            base of operation description.
        configure_kw: dict[str, dict[str, Any]]
            parameters for update standard settings from MessageFormat.
        set_kw: dict[str, Any]
            message content for .set method.

        Returns
        -------
        Message
            filled message with specified message format.

        Raises
        ------
        AttributeError
            if message format is not specified.
        """
        if self.mf is None:
            raise AttributeError("message format not specified")  # todo: custom exception
        msg = self.mf.get(**configure_kw)

        set_kw["address"] = self.address
        if operation_base != "w" and "data_length" not in set_kw:
            set_kw["data_length"] = self.length
        if "operation" not in set_kw:
            set_kw["operation"] = self._find_operation(msg, operation_base)

        return msg.set(**set_kw)

    @classmethod
    def from_series(
            cls, series: pd.Series, mf: MessageFormat = None
    ) -> Register:
        """
        Get register from pandas.Series

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
        return pd.Series(dict(
            external_name=self.external_name,
            name=self.name,
            format_name=self.format_name,
            address=self.address,
            register_type=self.register_type,
            length=self.length,
            data__fmt=self.data__fmt,
            description=self.description,
        ))

    def __add__(self, shift: int) -> Register:
        """Shift the address by a specified number."""
        return self.shift(shift)


class RegisterMap(object):
    """
    Represents class instance to store registers.

    Parameters
    ----------
    registers: pandas.DataFrame
        table with parameters for registers.
    """

    EXPECTED_COLUMNS = (
        "external_name",
        "name",
        "format_name",
        "address",
        "register_type",
        "length",
        "data__fmt",
        "description",
    )
    "tuple of expected columns in `register_table`"

    def __init__(self, registers: pd.DataFrame):
        self._tbl = self._validate_table(registers)

    def get(self, name: str, pf: PackageFormat = None) -> Register:
        """
        Get register by name.

        Searches the register first in 'name' column, then, if not found,
        in 'external_name' column.

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
        founded_names = self._tbl[self._tbl["name"] == name]
        founded_ext_names = self._tbl[self._tbl["external_name"] == name]
        names_counts, ext_names_count = len(founded_names), len(founded_ext_names)

        if names_counts > 1 or ext_names_count > 1:
            raise ValueError(
                "several registers with name %s: %d in 'name', %d in "
                "'external name'" % (name, names_counts, ext_names_count)
            )
        elif names_counts + ext_names_count == 0:
            raise AttributeError("register %r not found" % name)

        series = (
            founded_names if names_counts else founded_ext_names
        ).iloc[0]

        if pf is None:
            mf = None
        else:
            mf = pf.get_format(series["format_name"])

        return Register.from_series(series, mf=mf)

    def write(
            self,
            database: str | Path | sqlite3.Connection,
            if_exists: str = "replace"
    ) -> None:
        """
        Write register map table to sqlite table.

        Parameters
        ----------
        database: str | Path | sqlite3.Connection
            path or connection to a database.
        if_exists: str
            How to behave if the table already exists.

            * fail: Raise a ValueError.
            * replace: Drop the table before inserting new values.
        """
        if if_exists == "append":
            raise ValueError("'append' not available for if_exists")
        if if_exists not in {"replace", "fail"}:
            raise ValueError(f"'{if_exists}' is not valid for if_exists")

        if isinstance(database, str | Path):
            with RWSQLite(database, autocommit=False) as rws:
                self._tbl.to_sql(
                    "registers",
                    rws.connection,
                    index=False,
                    if_exists=if_exists
                )
            return
        self._tbl.to_sql(
            "registers", database, index=False, if_exists=if_exists
        )

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
        ref, cols = set(self.EXPECTED_COLUMNS), set(table.columns)
        invalid_cols, missing_cols = cols - ref, ref - cols
        if len(missing_cols):
            raise ValueError(f"missing columns: {missing_cols}")
        if len(invalid_cols):
            raise ValueError(f"invalid columns: {invalid_cols}")  # todo: custom exc
        # todo: checks:
        #  without dublicates in external_name and name
        #  without duplicates in address with the same msg_fmt
        #  addresses crossing by length?
        return table.sort_values(by=["format_name", "address"])

    @classmethod
    def read(cls, database: str | Path | sqlite3.Connection) -> RegisterMap:
        """
        Read RegisterMap from database.

        Data will be readed from 'registers' table.

        Parameters
        ----------
        database: Path | sqlite3.Connection
            path or connection to the database.

        Returns
        -------
        RegisterMap
            register map instance.
        """
        if isinstance(database, str | Path):
            with RWSQLite(database, autocommit=False) as rws:
                return RegisterMap(pd.read_sql(
                    "SELECT * FROM registers", rws.connection
                ))
        return RegisterMap(pd.read_sql("SELECT * FROM registers", database))

    @property
    def table(self) -> pd.DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            table with parameters of registers.
        """
        return self._tbl

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



"""Private module of ``pyiak_instr.communication.format_map.types``."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from itertools import takewhile
from abc import ABC
from typing import Any, Generic, Self, TypeVar

import pandas as pd

from ....core import Code
from ....exceptions import NotAmongTheOptions
from ....types import SubPatternAdditions
from ....store.bin.types import STRUCT_DATACLASS
from ...message.types import MessagePatternABC, MessageABC


__all__ = ["RegisterABC", "RegistersMapABC"]


MessageT = TypeVar("MessageT", bound=MessageABC[Any, Any, Any, Any])
RegisterT = TypeVar(
    "RegisterT", bound="RegisterABC[MessageABC[Any, Any, Any, Any]]"
)


@STRUCT_DATACLASS
class RegisterABC(ABC, Generic[MessageT]):
    """
    Base structure for device/service register.
    """

    pattern: str
    "message format name."

    name: str
    "the name of the register"

    address: int
    "register address. Used for address field in message."

    length: int
    "register length in bytes."

    rw_type: Code = Code.ANY
    "register type"

    description: str = ""
    "register description. First sentence must be a short summary."

    def __post_init__(self) -> None:
        if self.rw_type not in {
            Code.ANY,
            Code.READ_ONLY,
            Code.WRITE_ONLY,
        }:
            raise NotAmongTheOptions(
                "rw_type",
                self.rw_type,
                {Code.ANY, Code.READ_ONLY, Code.WRITE_ONLY},
            )

    def get(
        self,
        pattern: MessagePatternABC[MessageT, Any],
        changes_allowed: bool = True,
        sub_additions: SubPatternAdditions = SubPatternAdditions(),
        top_additions: dict[str, Any] | None = None,
        fields_data: dict[str, Any] | None = None,
        autoupdate_fields: bool = True,
        operation: Code | None = None,
        dynamic_length: int = 0,
        data: Any = None,
    ) -> MessageT:
        """
        Get message from `pattern`.

        Parameters
        ----------
        pattern : MessagePatternABC[MessageT, Any]
            pattern for message.
        changes_allowed : bool
            indicates that changes pattern is allowed when cheating new
            instance.
        sub_additions : SubPatternAdditions, default=SubPatternAdditions()
            additions for sub-pattern.
        top_additions : dict[str, Any] | None, default=None
            additions for `pattern`.
        fields_data: dict[str, Any] | None, default=None
            data for fields.
        autoupdate_fields: bool, default=True
            autoupdate field content if it is possible.
        operation : Code | None, default=None
            operation code.
        dynamic_length : int, default=0
            length of dynamic field. Works only if dynamic field and dynamic
            length field exists.
        data : Any, default=None
            content of dynamic field.

        Returns
        -------
        MessageT
            message instance.

        Raises
        ------
        AttributeError
            if pattern not specified (is None).
        """
        if top_additions is None:
            top_additions = {}
        if fields_data is None:
            fields_data = {}

        msg: MessageT = pattern.get_for_direction(
            Code.TX,
            changes_allowed=changes_allowed,
            sub_additions=sub_additions,
            **top_additions,
        )

        if msg.has.address:
            fields_data[msg.get.address.name] = self.address

        if msg.struct.is_dynamic:
            if data is not None:
                fields_data[msg.struct.dynamic_field_name] = data

            if dynamic_length > 0 and msg.has.dynamic_length:
                fields_data[msg.get.dynamic_length.name] = dynamic_length

        if operation is not None and msg.has.operation:
            fields_data[msg.get.operation.name] = operation

        msg.encode(**fields_data)
        if autoupdate_fields:
            msg.autoupdate_fields()

        return msg

    def read(
        self,
        pattern: MessagePatternABC[MessageT, Any],
        dynamic_length: int = 0,
        changes_allowed: bool = True,
        sub_additions: SubPatternAdditions = SubPatternAdditions(),
        top_additions: dict[str, Any] | None = None,
        fields_data: dict[str, Any] | None = None,
        autoupdate_fields: bool = True,
    ) -> MessageT:
        """
        Get message from `pattern` with READ operation.

        Parameters
        ----------
        pattern : MessagePatternABC[MessageT, Any]
            pattern for message.
        dynamic_length : int, default=0
            length of dynamic field. Works only if dynamic field and dynamic
            length field exists.
        changes_allowed : bool
            indicates that changes pattern is allowed when cheating new
            instance.
        sub_additions : SubPatternAdditions, default=SubPatternAdditions()
            additions for sub-pattern.
        top_additions : dict[str, Any] | None, default=None
            additions for `pattern`.
        fields_data: dict[str, Any] | None, default=None
            data for fields.
        autoupdate_fields: bool, default=True
            autoupdate field content if it is possible.

        Returns
        -------
        MessageT
            message instance.
        """
        if dynamic_length <= 0:
            dynamic_length = self.length
        return self.get(
            pattern,
            changes_allowed=changes_allowed,
            sub_additions=sub_additions,
            top_additions=top_additions,
            fields_data=fields_data,
            autoupdate_fields=autoupdate_fields,
            operation=Code.READ,
            dynamic_length=dynamic_length,
        )

    def write(
        self,
        pattern: MessagePatternABC[MessageT, Any],
        data: Any,
        changes_allowed: bool = True,
        sub_additions: SubPatternAdditions = SubPatternAdditions(),
        top_additions: dict[str, Any] | None = None,
        fields_data: dict[str, Any] | None = None,
        autoupdate_fields: bool = True,
    ) -> MessageT:
        """
        Get message from `pattern`.

        Parameters
        ----------
        pattern : MessagePatternABC[MessageT, Any]
            pattern for message.
        data : Any, default=None
            content of dynamic field.
        changes_allowed : bool
            indicates that changes pattern is allowed when cheating new
            instance.
        sub_additions : SubPatternAdditions, default=SubPatternAdditions()
            additions for sub-pattern.
        top_additions : dict[str, Any] | None, default=None
            additions for `pattern`.
        fields_data: dict[str, Any] | None, default=None
            data for fields.
        autoupdate_fields: bool, default=True
            autoupdate field content if it is possible.

        Returns
        -------
        MessageT
            message instance.
        """
        return self.get(
            pattern,
            changes_allowed=changes_allowed,
            sub_additions=sub_additions,
            top_additions=top_additions,
            fields_data=fields_data,
            autoupdate_fields=autoupdate_fields,
            operation=Code.WRITE,
            data=data,
        )

    @classmethod
    def from_series(cls, series: pd.Series[Any]) -> Self:
        """
        Initialize class instance via pandas series.

        None values will be dropped.

        Parameters
        ----------
        series : pd.Series[Any]
            series with data.

        Returns
        -------
        Self
            initialized self instance.
        """
        series_dict: dict[str, Any] = series.dropna().to_dict()
        if "rw_type" in series_dict:
            series_dict["rw_type"] = Code(series["rw_type"])
        return cls(**series_dict)

    @property
    def series(self) -> pd.Series[Any]:
        """
        Returns
        -------
        pd.Series[Any]
            series with data from dataclass.
        """
        return pd.Series(self.__init_kwargs__())

    @property
    def short_description(self) -> str:
        """
        Returns
        -------
        str
            first sentence of `description`.
        """
        return "".join(takewhile(lambda c: c != ".", self.description))

    def __init_kwargs__(self) -> dict[str, Any]:
        return dict(
            pattern=self.pattern,
            name=self.name,
            address=self.address,
            length=self.length,
            rw_type=self.rw_type,
            description=self.description,
        )


class RegistersMapABC(ABC, Generic[RegisterT]):
    """
    Base class for store registers.

    Parameters
    ----------
    table : pd.DataFrame
        table with data for registers.
    """

    _register_type: type[RegisterT]

    _register_columns: set[str] = {
        "pattern",
        "name",
        "address",
        "length",
        "rw_type",
    }

    _required_columns: set[str] = set()

    def __init__(self, table: pd.DataFrame) -> None:
        for col in self._register_columns:
            self._required_columns.add(col)
        self._verify_table(table)
        self._table = table

    def get_register(
        self,
        name: str,
    ) -> RegisterT:
        """
        Get register by `name`.

        Parameters
        ----------
        name : str
            name of the register.

        Returns
        -------
        RegisterT
            register instance.

        Raises
        ------
        ValueError
            if register with `name` not found;
            if `table` have more than one register with `name`.
        """
        reg_table = self._table[self._table["name"] == name]
        if reg_table.shape[0] == 0:
            raise ValueError(f"register with name '{name}' not found")
        if reg_table.shape[0] > 1:
            raise ValueError(
                f"there is more than one register with the name '{name}'"
            )

        return self._register_type.from_series(
            reg_table[list(self._register_columns)].iloc[0],
        )

    def write(self, path: Path) -> None:
        """
        Write registers table to `path`.

        Parameters
        ----------
        path : Path
            path to database.
        """
        with sqlite3.connect(path) as con:
            self._table.to_sql(
                "registers", con, index=False, if_exists="replace"
            )

    def _verify_table(self, table: pd.DataFrame) -> None:
        """
        Verify table with registers data.

        Parameters
        ----------
        table : pd.DataFrame
            table with registers data.

        Raises
        ------
        ValueError
            if there is at least one required column in the table.
        """
        diff = self._required_columns - set(table.columns)
        if len(diff) > 0:
            raise ValueError(f"missing columns in table: {diff}")

    @classmethod
    def from_registers(cls, *registers: RegisterT) -> Self:
        """
        Get class instance from registers.

        Parameters
        ----------
        *registers : RegisterT
            register instances.

        Returns
        -------
        Self
            self instance.
        """
        return cls.from_series(*(r.series for r in registers))

    @classmethod
    def from_series(cls, *series: pd.Series[Any]) -> Self:
        """
        Get class instance from series.

        Parameters
        ----------
        *series : pd.Series[Any]
            series instances.

        Returns
        -------
        Self
            self instance.
        """
        return cls(pd.DataFrame(data=series))

    @classmethod
    def read(cls, path: Path) -> Self:
        """
        Get class instance which initialized from sql database.

        Parameters
        ----------
        path : Path
            path to database.

        Returns
        -------
        Self
            initialized self instance.
        """
        with sqlite3.connect(path) as con:
            return cls(pd.read_sql("SELECT * FROM registers", con))

    @property
    def table(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            table with registers data.
        """
        return self._table

"""Private module of ``pyiak_instr.communication.format_map.types``."""
from __future__ import annotations
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

    pattern: MessagePatternABC[MessageT, Any] | None = None
    "message format for messages of this register."

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
        if self.pattern is None:
            raise AttributeError("pattern not specified")

        if top_additions is None:
            top_additions = {}
        if fields_data is None:
            fields_data = {}

        msg: MessageT = self.pattern.get(
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
            changes_allowed=changes_allowed,
            sub_additions=sub_additions,
            top_additions=top_additions,
            fields_data=fields_data,
            autoupdate_fields=autoupdate_fields,
            operation=Code.WRITE,
            data=data,
        )

    @classmethod
    def from_series(
        cls,
        series: pd.Series[Any],
        pattern: MessagePatternABC[MessageT, Any] | None = None,
    ) -> Self:
        """
        Initialize class instance via pandas series.

        None values will be dropped.

        Parameters
        ----------
        series : pd.Series[Any]
            series with data.
        pattern : MessagePatternABC[MessageT, Any] | None, default=None
            pattern for message.

        Returns
        -------
        Self
            initialized self instance.
        """
        series_dict: dict[str, Any] = series.dropna().to_dict()
        if "rw_type" in series_dict:
            series_dict["rw_type"] = Code(series["rw_type"])
        return cls(
            pattern=pattern,  # type: ignore[call-arg]
            **series_dict,
        )

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

    _register_columns: set[str] = {"name", "address", "length", "rw_type"}

    _required_columns: set[str] = {"pattern"}

    def __init__(self, table: pd.DataFrame) -> None:
        for col in self._register_columns:
            self._required_columns.add(col)
        self._verify_table(table)
        self._table = table

    def get_register(
        self,
        name: str,
        pattern: MessagePatternABC[Any, Any] | None = None,
    ) -> RegisterT:
        """
        Get register by `name`.

        Parameters
        ----------
        name : str
            name of the register.
        pattern : MessagePatternABC[Any, Any] | None, default=None
            pattern for message instance.

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
            pattern=pattern,
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

    @property
    def table(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            table with registers data.
        """
        return self._table

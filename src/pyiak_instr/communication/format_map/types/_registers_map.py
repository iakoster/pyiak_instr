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


__all__ = ["RegisterStructABC"]


MessageT = TypeVar("MessageT", bound=MessageABC[Any, Any, Any, Any])


@STRUCT_DATACLASS
class RegisterStructABC(ABC, Generic[MessageT]):
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

        msg.encode(**fields_data)
        if autoupdate_fields:
            msg.autoupdate_fields()
        # todo: check data length and register length

        return msg

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
        return cls(
            pattern=pattern,  # type: ignore[call-arg]
            **series.dropna().to_dict(),
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

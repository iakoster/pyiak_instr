"""Private module of ``pyiak_instr.communication.message``"""
from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

from .types import (
    MessageFieldStructABCUnionT,
    MessageStructABC,
    MessageABC,
)

if TYPE_CHECKING:
    from ._pattern import MessagePattern as _MessagePattern


__all__ = ["Message"]


AddressT = TypeVar("AddressT")
MessagePattern: TypeAlias = "_MessagePattern"


class Message(
    MessageABC[
        MessageFieldStructABCUnionT,
        MessageStructABC[MessageFieldStructABCUnionT],
        MessagePattern,
        AddressT,
    ],
    Generic[AddressT],
):
    """
    Message for communication between devices.
    """

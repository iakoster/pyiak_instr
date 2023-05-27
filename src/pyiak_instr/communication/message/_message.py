# """Private module of ``pyiak_instr.communication.message`` with message
# classes."""
from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar

from .types import (
    MessageFieldStructABCUnionT,
    MessageStructABC,
    MessageABC,
)

if TYPE_CHECKING:
    from ._pattern import MessagePattern


__all__ = ["Message"]


AddressT = TypeVar("AddressT")


class Message(
    MessageABC[
        MessageFieldStructABCUnionT,
        MessageStructABC,
        "MessagePattern",
        AddressT,
    ],
    Generic[AddressT],
):
    """
    Message for communication between devices.
    """

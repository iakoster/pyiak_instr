from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from ....types.store.bin import (
    BytesFieldStructPatternABC,
    BytesStorageStructPatternABC,
    BytesStoragePatternABC,
)
from ._struct import MessageFieldStructABC, MessageStructABC
from ._message import MessageABC


__all__ = [
    "MessageFieldStructPatternABC",
    "MessageStructPatternABC",
    "MessagePatternABC",
]


FieldStructT = TypeVar("FieldStructT", bound=MessageFieldStructABC)
MessageStructT = TypeVar(
    "MessageStructT", bound=MessageStructABC[MessageFieldStructABC]
)
MessageT = TypeVar("MessageT", bound=MessageABC)

FieldStructPatternT = TypeVar(
    "FieldStructPatternT", bound="MessageFieldStructPatternABC"
)
MessageStructPatternT = TypeVar(
    "MessageStructPatternT", bound="MessageStructPatternABC"
)


class MessageFieldStructPatternABC(BytesFieldStructPatternABC[FieldStructT]):
    ...


class MessageStructPatternABC(
    BytesStorageStructPatternABC[MessageStructT, FieldStructPatternT]
):
    ...


class MessagePatternABC(
    BytesStoragePatternABC[MessageT, MessageStructPatternT]
):
    ...

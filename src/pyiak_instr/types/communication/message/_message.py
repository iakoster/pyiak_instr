"""Private module of ``pyiak_instr.types.communication`` with types for
communication module."""
from abc import ABC, abstractmethod
from dataclasses import field as _field
from functools import wraps
from typing import (  # pylint: disable=unused-import
    Any,
    Callable,
    Generator,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from ....core import Code
from ....types import Encoder
from ....types.store.bin import (
    BytesDecodeT,
    BytesEncodeT,
    BytesStorageABC,
)
from ._struct import (
    STRUCT_DATACLASS,
    MessageFieldStructABC,
    MessageStructGetParserABC,
    MessageStructHasParserABC,
    MessageStructABC,
)


__all__ = [
    "MessageABC",
]


AddressT = TypeVar("AddressT")
FieldStructT = TypeVar("FieldStructT", bound=MessageFieldStructABC)
MessageStructT = TypeVar(
    "MessageStructT", bound=MessageStructABC[MessageFieldStructABC]
)
MessagePatternT = TypeVar("MessagePatternT")

# StructT = TypeVar("StructT") # , bound=BytesFieldStructProtocol)
# FieldT = TypeVar("FieldT", bound="MessageFieldABC[Any, Any]")
# FieldAnotherT = TypeVar("FieldAnotherT", bound="MessageFieldABC[Any, Any]")
# MessageGetParserT = TypeVar(
#     "MessageGetParserT", bound="MessageGetParserABC[Any, Any]"
# )
# MessageHasParserT = TypeVar(
#     "MessageHasParserT", bound="MessageHasParserABC[Any]"
# )
# MessageT = TypeVar(
#     "MessageT", bound="MessageABC[Any, Any, Any, Any, Any, Any]"
# )
# FieldPatternT = TypeVar("FieldPatternT", bound="MessageFieldPatternABC[Any]")
# MessagePatternT = TypeVar(
#     "MessagePatternT", bound="MessagePatternABC[Any, Any]"
# )


# todo: clear src and dst?
# todo: get rx and tx class instance
# todo: field parser
class MessageABC(
    BytesStorageABC[FieldStructT, MessageStructT, MessagePatternT],
    Generic[FieldStructT, MessageStructT, MessagePatternT, AddressT],
):

    def __init__(
            self,
            storage: MessageStructT,
            pattern: MessagePatternT | None = None,
    ):
        super().__init__(storage, pattern=pattern)
        self._src, self._dst = None, None

    def split(self) -> Generator[Self, None, None]:
        """
        Split the message into parts over an infinite field.

        Yields
        ------
        Self
            message part.
        """

    @property
    def dst(self) -> AddressT | None:
        """
        Returns
        -------
        AddressT | None
            destination address.
        """
        return self._dst

    @dst.setter
    def dst(self, destination: AddressT | None) -> None:
        """
        Set destination address.

        Parameters
        ----------
        destination : AddressT | None
            destination address.
        """
        self._dst = destination

    @property
    def get(self) -> MessageStructGetParserABC[MessageStructT, FieldStructT]:
        """
        Returns
        -------
        MessageStructGetParserABC
            message struct get parser.
        """
        return self._s.get

    @property
    def has(self) -> MessageStructHasParserABC[MessageStructT, FieldStructT]:
        """
        Returns
        -------
        MessageStructHasParserABC
            message struct has parser.
        """
        return self._s.has

    @property
    def src(self) -> AddressT | None:
        """
        Returns
        -------
        AddressT | None
            source address.
        """
        return self._src

    @src.setter
    def src(self, source: AddressT | None) -> None:
        """
        Set source address.

        Parameters
        ----------
        source : AddressT | None
            source address.
        """
        self._src = source


# class MessageFieldPatternABC(Generic[StructT]):  # BytesFieldPatternABC[StructT]
#     """
#     Represent abstract class of pattern for message field.
#     """
#
#     @staticmethod
#     @abstractmethod
#     def get_bytesize(fmt: Code) -> int:
#         """
#         Get fmt size in bytes.
#
#         Parameters
#         ----------
#         fmt : Code
#             fmt code.
#
#         Returns
#         -------
#         int
#             fmt bytesize.
#         """
#
#
# class MessagePatternABC(
#     # ContinuousBytesStoragePatternABC[MessageT, FieldPatternT],
#     Generic[MessageT, FieldPatternT],
# ):
#     """
#     Represent abstract class of pattern for message.
#     """

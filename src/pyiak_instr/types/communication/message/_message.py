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

from src.pyiak_instr.exceptions import ContentError, NotAmongTheOptions
from src.pyiak_instr.core import Code
from src.pyiak_instr.types._encoders import Encoder
from src.pyiak_instr.types.store.bin import (
    STRUCT_DATACLASS,
    BytesDecodeT,
    BytesEncodeT,
    BytesFieldStructABC,
    BytesFieldStructPatternABC,
    BytesStorageABC,
    BytesStoragePatternABC,
    BytesStorageStructABC,
    BytesStorageStructPatternABC,
    ContinuousBytesStorageStructPatternABC,
)


__all__ = [
    "MessageFieldStructABC",
    "SingleMessageFieldStructABC",
    "StaticMessageFieldStructABC",
    "AddressMessageFieldStructABC",
    "CrcMessageFieldStructABC",
    "DataMessageFieldStructABC",
    "DataLengthMessageFieldStructABC",
    "IdMessageFieldStructABC",
    "OperationMessageFieldStructABC",
    "ResponseMessageFieldStructABC",
    "MessageStructABC",
    "MessageABC",
]


AddressT = TypeVar("AddressT")
EncoderT: TypeAlias = Encoder[BytesDecodeT, BytesEncodeT, bytes]
FieldStructT = TypeVar("FieldStructT", bound="MessageFieldStructABC")
MessageStructT = TypeVar("MessageStructT", bound="MessageStructABC")
MessageT = TypeVar("MessageT", bound="MessageABC")
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
class MessageABC(
    BytesStorageABC[FieldStructT, MessageStructT, MessagePatternT]
):

    _src: AddressT | None = None
    """Source address."""

    _dst: AddressT | None = None
    """Destination address."""

    def __init__(
            self,
            storage: MessageStructT,
            pattern: MessagePatternT | None = None,
    ):
        super().__init__(storage, pattern=pattern)

        self._field_types: dict[Code, str] = {}
        for struct in self.struct:
            field_type = struct.field_type
            if field_type not in self._field_types:
                self._field_types[field_type] = struct.name

    def get(self, type_: Code) -> str:
        """
        Returns
        -------
        str
            field name with specified type.
        """
        if type_ not in self._field_types:
            raise TypeError(f"field instance with type {type_!r} not found")
        return self._field_types[type_]

    def has(self, type_: Code) -> bool:
        """
        Returns
        -------
        bool
            True - message has field of specified type.
        """
        return type_ in self._field_types

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

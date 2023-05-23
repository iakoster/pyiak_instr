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


class MessageGetParserABC(Generic[FieldStructT]):

    def __init__(
            self,
            type_codes: dict[Code, type[FieldStructT]],
            types: dict[type[FieldStructT], str],
    ) -> None:
        self._codes = type_codes
        self._types = types

    @property
    def basic(self) -> str:
        return self(Code.BASIC)

    @property
    def single(self) -> str:
        return self(Code.SINGLE)

    @property
    def static(self) -> str:
        return self(Code.STATIC)

    @property
    def address(self) -> str:
        return self(Code.ADDRESS)

    @property
    def crc(self) -> str:
        return self(Code.CRC)

    @property
    def data(self) -> str:
        return self(Code.DATA)

    @property
    def data_length(self) -> str:
        return self(Code.DATA_LENGTH)

    @property
    def id_(self) -> str:
        return self(Code.ID)

    @property
    def operation(self) -> str:
        return self(Code.OPERATION)

    @property
    def response(self) -> str:
        return self(Code.RESPONSE)

    def __call__(self, code: Code) -> str:
        if code not in self._codes:
            raise TypeError(f"undefined code: {code!r}")
        type_ = self._codes[code]
        if type_ not in self._types:
            raise TypeError(
                f"field instance with type {type_.__name__!r} "
                "not found"
            )
        return self._types[type_]


class MessageHasParserABC(Generic[FieldStructT]):

    def __init__(
            self,
            type_codes: dict[Code, type[FieldStructT]],
            types: dict[type[FieldStructT], str],
    ) -> None:
        self._codes = type_codes
        self._types = types

    @property
    def basic(self) -> bool:
        return self(Code.BASIC)

    @property
    def single(self) -> bool:
        return self(Code.SINGLE)

    @property
    def static(self) -> bool:
        return self(Code.STATIC)

    @property
    def address(self) -> bool:
        return self(Code.ADDRESS)

    @property
    def crc(self) -> bool:
        return self(Code.CRC)

    @property
    def data(self) -> bool:
        return self(Code.DATA)

    @property
    def data_length(self) -> bool:
        return self(Code.DATA_LENGTH)

    @property
    def id_(self) -> bool:
        return self(Code.ID)

    @property
    def operation(self) -> bool:
        return self(Code.OPERATION)

    @property
    def response(self) -> bool:
        return self(Code.RESPONSE)

    def __call__(self, code: Code) -> bool:
        return code in self._codes and self._codes[code] in self._types


# todo: clear src and dst?
# todo: get rx and tx class instance
# todo: field parser
class MessageABC(
    BytesStorageABC[FieldStructT, MessageStructT, MessagePatternT],
    Generic[FieldStructT, MessageStructT, MessagePatternT, AddressT],
):

    _field_codes: dict[Code, type[FieldStructT]]

    def __init__(
            self,
            storage: MessageStructT,
            pattern: MessagePatternT | None = None,
    ):
        super().__init__(storage, pattern=pattern)
        self._src, self._dst = None, None

        self._field_types: dict[type[FieldStructT], str] = {}
        for struct in self.struct:
            field_class = struct.__class__
            if field_class not in self._field_types:
                self._field_types[field_class] = struct.name

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
    def get(self) -> MessageGetParserABC[FieldStructT]:
        """
        Returns
        -------
        MessageGetParserABC
            message get parser.
        """
        return MessageGetParserABC(self._field_codes, self._field_types)

    @property
    def has(self) -> MessageHasParserABC[FieldStructT]:
        """
        Returns
        -------
        MessageHasParserABC
            message has parser.
        """
        return MessageHasParserABC(self._field_codes, self._field_types)

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

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

from ...exceptions import ContentError, NotAmongTheOptions
from ...core import Code
from .._encoders import Encoder
from ..store.bin import (
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


# todo: refactor (join classes to one and use metaclass)
@STRUCT_DATACLASS
class MessageFieldStructABC(BytesFieldStructABC):
    ...


@STRUCT_DATACLASS
class SingleMessageFieldStructABC(MessageFieldStructABC):

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        if self.words_expected != 1:
            raise ValueError(
                f"{self.__class__.__name__} should expect one word"
            )

    def _modify_values(self) -> None:
        if self.stop is None and self.bytes_expected == 0:
            object.__setattr__(self, "bytes_expected", self.word_bytesize)
        super()._modify_values()


@STRUCT_DATACLASS
class StaticMessageFieldStructABC(SingleMessageFieldStructABC):

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        if not self.has_default:
            raise ValueError("default value not specified")

    def verify(self, content: bytes, raise_if_false: bool = False) -> bool:
        correct = super().verify(content, raise_if_false=raise_if_false)
        if correct:
            correct = content == self.default
            if not correct and raise_if_false:
                raise ContentError(self, clarification=content.hex(" "))
        return correct


@STRUCT_DATACLASS
class AddressMessageFieldStructABC(SingleMessageFieldStructABC):

    behaviour: Code = Code.DMA

    units: Code = Code.WORDS

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        if self.behaviour not in {Code.DMA, Code.STRONG}:
            raise NotAmongTheOptions(
                "behaviour", self.behaviour, {Code.DMA, Code.STRONG}
            )
        if self.units not in {Code.BYTES, Code.WORDS}:
            raise NotAmongTheOptions(
                "units", self.units, {Code.BYTES, Code.WORDS}
            )


@STRUCT_DATACLASS
class CrcMessageFieldStructABC(SingleMessageFieldStructABC):

    poly: int = 0x1021

    init: int = 0

    wo_fields: set[str] = _field(default_factory=set)

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        if self.bytes_expected != 2 or self.poly != 0x1021 or self.init != 0:
            raise NotImplementedError(
                "Crc algorithm not verified for other values"
            )  # todo: implement for any crc

    def calculate(self, content: bytes) -> int:
        """
        Calculate crc of content.

        Parameters
        ----------
        content : bytes
            content to calculate crc.

        Returns
        -------
        int
            crc value of `content`.
        """

        crc = self.init
        for byte in content:
            crc ^= byte << 8
            for _ in range(8):
                crc <<= 1
                if crc & 0x10000:
                    crc ^= self.poly
                crc &= 0xFFFF
        return crc


@STRUCT_DATACLASS
class DataMessageFieldStructABC(MessageFieldStructABC):
    """Represents a field of a Message with data."""


@STRUCT_DATACLASS
class DataLengthMessageFieldStructABC(SingleMessageFieldStructABC):

    behaviour: Code = Code.ACTUAL  # todo: logic
    """determines the behavior of determining the content value."""

    units: Code = Code.BYTES
    """data length units. Data can be measured in bytes or words."""

    additive: int = 0
    """additional value to the length of the data."""

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        if self.additive < 0:
            raise ValueError("additive number must be positive integer")
        if self.behaviour not in {Code.ACTUAL, Code.EXPECTED}:
            raise NotAmongTheOptions("behaviour", self.behaviour)
        if self.units not in {Code.BYTES, Code.WORDS}:
            raise NotAmongTheOptions("units", self.units)

    def calculate(self, data: bytes, value_size: int) -> int:
        if self.units is Code.WORDS:
            if len(data) % value_size != 0:
                raise ContentError(self, "non-integer words count in data")
            return len(data) // value_size

        # units is a BYTES
        return len(data)


@STRUCT_DATACLASS
class IdMessageFieldStructABC(SingleMessageFieldStructABC):
    """Represents a field with a unique identifier of a particular message."""


@STRUCT_DATACLASS
class OperationMessageFieldStructABC(SingleMessageFieldStructABC):
    descs: dict[int, Code] = _field(
        default_factory=lambda: {0: Code.READ, 1: Code.WRITE}
    )
    """matching dictionary value and codes."""

    descs_r: dict[Code, int] = _field(init=False)
    """reversed `descriptions`."""

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        self._setattr("descs_r", {v: k for k, v in self.descs.items()})

    def encode(
            self, content: BytesEncodeT | Code, verify: bool = False
    ) -> bytes:
        if isinstance(content, Code):
            value = self.desc_r(content)
            if value is None:
                raise ContentError(self, f"can't encode {repr(content)}")
            content = value
        return super().encode(content, verify=True)

    def desc(self, value: int) -> Code:
        # pylint: disable=unsupported-membership-test,unsubscriptable-object
        if value not in self.descs:
            return Code.UNDEFINED
        return self.descs[value]

    def desc_r(self, code: Code) -> int | None:
        if code not in self.descs_r:
            return None
        return self.descs_r[code]

    def _modify_values(self) -> None:
        super()._modify_values()
        self._setattr("descs_r", {v: k for k, v in self.descs.items()})


@STRUCT_DATACLASS
class ResponseMessageFieldStructABC(SingleMessageFieldStructABC):
    descs: dict[int, Code] = _field(default_factory=dict)
    """matching dictionary value and codes."""

    descs_r: dict[Code, int] = _field(init=False)
    """reversed `descriptions`."""

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        self._setattr("descs_r", {v: k for k, v in self.descs.items()})

    def encode(
            self, content: BytesEncodeT | Code, verify: bool = False
    ) -> bytes:
        if isinstance(content, Code):
            value = self.desc_r(content)
            if value is None:
                raise ContentError(self, f"can't encode {repr(content)}")
            content = value
        return super().encode(content)

    def desc(self, value: int) -> Code:
        """
        Convert value to code.

        Returns `UNDEFINED` if value not represented in `descs`.

        Parameters
        ----------
        value : int
            value for converting.

        Returns
        -------
        Code
            value code.
        """
        # pylint: disable=unsupported-membership-test,unsubscriptable-object
        if value not in self.descs:
            return Code.UNDEFINED
        return self.descs[value]

    def desc_r(self, code: Code) -> int | None:
        """
        Convert code to value.

        Returns None if `code` not represented in `descs`.

        Parameters
        ----------
        code : Code
            code for converting.

        Returns
        -------
        int | None
            code value.
        """
        if code not in self.descs_r:
            return None
        return self.descs_r[code]


@STRUCT_DATACLASS
class MessageStructABC(BytesStorageStructABC[FieldStructT]):

    divisible: bool = False
    """shows that the message can be divided by the infinite field."""

    mtu: int = 1500
    """max size of one message part."""

    def __post_init__(self, fields: dict[str, FieldStructT]) -> None:
        super().__post_init__(fields)
        if self.divisible:
            if not self.is_dynamic:
                raise TypeError(
                    f"{self.__class__.__name__} can not be divided because "
                    "it does not have a dynamic field"
                )

            min_mtu = (
                self.minimum_size + self._f[self._dyn_field].word_bytesize
            )
            if self.mtu < min_mtu:
                raise ValueError(
                    "MTU value does not allow you to split the message if "
                    f"necessary. The minimum MTU is {min_mtu} "
                    f"(got {self.mtu})"
                )


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

"""Private module of ``pyiak_instr.communication.message`` with field
structs."""
from __future__ import annotations
from dataclasses import dataclass, field as _field
from typing import ClassVar, Self, Union

from ...core import Code
from ...exceptions import NotAmongTheOptions
from ...types import PatternABC
from ...store import BytesFieldStruct


__all__ = [
    "MessageFieldStruct",
    "SingleMessageFieldStruct",
    "StaticMessageFieldStruct",
    "AddressMessageFieldStruct",
    "CrcMessageFieldStruct",
    "DataMessageFieldStruct",
    "DataLengthMessageFieldStruct",
    "IdMessageFieldStruct",
    "OperationMessageFieldStruct",
    "ResponseMessageFieldStruct",
    "MessageFieldStructUnionT",
    "MessageFieldPattern",
]


@dataclass(frozen=True, kw_only=True)
class MessageFieldStruct(BytesFieldStruct):
    """Represents a general field of a Message."""


@dataclass(frozen=True, kw_only=True)
class SingleMessageFieldStruct(MessageFieldStruct):
    """
    Represents a field of a Message with single word.
    """

    def __post_init__(self) -> None:
        if self.stop is None and self.bytes_expected == 0:
            object.__setattr__(self, "bytes_expected", self.word_bytesize)
        super().__post_init__()
        if self.words_expected != 1:
            raise ValueError("single field should expect one word")


@dataclass(frozen=True, kw_only=True)
class StaticMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with static single word (e.g. preamble).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.has_default:
            raise ValueError("default value not specified")

    def verify(self, content: bytes) -> bool:
        """
        Verify the content for compliance with the field parameters.

        Also checks that `content` equal to default.

        Parameters
        ----------
        content: bytes
            content for validating.

        Returns
        -------
        bool
            True - content is correct, False - not.
        """
        if super().verify(content):
            return content == self.default
        return False


@dataclass(frozen=True, kw_only=True)
class AddressMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with address.
    """

    behaviour: Code = Code.DMA  # todo: logic

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.behaviour not in {Code.DMA, Code.STRONG}:
            raise NotAmongTheOptions(
                "behaviour", self.behaviour, {Code.DMA, Code.STRONG}
            )


@dataclass(frozen=True, kw_only=True)
class CrcMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with crc value.
    """

    poly: int = 0x1021

    init: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bytes_expected != 2 or self.poly != 0x1021 or self.init != 0:
            raise NotImplementedError(
                "Crc algorithm not verified for other values"
            )  # todo: optimize for any crc

    def get_crc(self, content: bytes) -> int:
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


@dataclass(frozen=True, kw_only=True)
class DataMessageFieldStruct(MessageFieldStruct):
    """Represents a field of a Message with data."""


@dataclass(frozen=True, kw_only=True)
class DataLengthMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with data length.
    """

    behaviour: Code = Code.ACTUAL  # todo: logic
    """determines the behavior of determining the content value."""

    units: Code = Code.BYTES
    """data length units. Data can be measured in bytes or words."""

    additive: int = 0
    """additional value to the length of the data."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.additive < 0:
            raise ValueError("additive number must be positive integer")
        if self.behaviour not in {Code.ACTUAL, Code.EXPECTED}:
            raise NotAmongTheOptions("behaviour", self.behaviour)
        if self.units not in {Code.BYTES, Code.WORDS}:
            raise NotAmongTheOptions("units", self.units)


@dataclass(frozen=True, kw_only=True)
class IdMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field with a unique identifier of a particular message.
    """


@dataclass(frozen=True, kw_only=True)
class OperationMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with operation (e.g. read).

    Operation codes are needed to compare the operation when receiving
    a message and generally to understand what operation is written in
    the message.

    If the dictionary is None, the standard value will be assigned
    {READ: 0, WRITE: 1}.
    """

    descriptions: dict[Code, int] = _field(
        default_factory=lambda: {Code.READ: 0, Code.WRITE: 1}
    )
    """dictionary of correspondence between the operation base and the value
    in the content."""

    _desc_r: ClassVar[dict[int, Code]] = {}
    """reversed `descriptions`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "_desc_r", {v: k for k, v in self.descriptions.items()}
        )


@dataclass(frozen=True, kw_only=True)
class ResponseMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with response field.
    """

    descriptions: dict[int, Code] = _field(default_factory=dict)
    """matching dictionary value and codes."""

    default_code: Code = Code.UNDEFINED
    """default code if value undefined."""


MessageFieldStructUnionT = Union[
    MessageFieldStruct,
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DataLengthMessageFieldStruct,
    IdMessageFieldStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
]


class MessageFieldPattern(PatternABC[MessageFieldStructUnionT]):
    """
    Represents pattern for message field struct
    """

    _options = {
        "basic": MessageFieldStruct,
        "single": SingleMessageFieldStruct,
        "static": StaticMessageFieldStruct,
        "address": AddressMessageFieldStruct,
        "crc": CrcMessageFieldStruct,
        "data": DataMessageFieldStruct,
        "data_length": DataLengthMessageFieldStruct,
        "id": IdMessageFieldStruct,
        "operation": OperationMessageFieldStruct,
        "response": ResponseMessageFieldStruct,
    }

    @classmethod
    def basic(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        stop: int | None = None,
        bytes_expected: int = 0,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for basic field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        stop : int | None, default=None
            index of stop byte.
        bytes_expected : int, default=0
            expected count of bytes.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename="basic",
            fmt=fmt,
            order=order,
            stop=stop,
            bytes_expected=bytes_expected,
            default=default,
        )

    @classmethod
    def single(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for single field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(typename="single", fmt=fmt, order=order, default=default)

    @classmethod
    def static(
        cls, fmt: Code, default: bytes, order: Code = Code.BIG_ENDIAN
    ) -> Self:
        """
        Get initialized pattern for static field.

        Parameters
        ----------
        fmt : Code
            value format.
        default : bytes
            default value for field.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(typename="static", fmt=fmt, order=order, default=default)

    @classmethod
    def address(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        behaviour: Code = Code.DMA,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for address field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        behaviour : Code, default=Code.DMA
            address field behaviour.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename="address",
            fmt=fmt,
            order=order,
            behaviour=behaviour,
            default=default,
        )

    @classmethod
    def crc(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        poly: int = 0x1021,
        init: int = 0,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for crc field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        poly : int, default=0x1021
            poly for crc algorithm.
        init : int, default=0
            init value for crc algorithm.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename="crc",
            fmt=fmt,
            order=order,
            poly=poly,
            init=init,
            default=default,
        )

    @classmethod
    def data(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        stop: int | None = None,
        bytes_expected: int = 0,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for data field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        stop : int | None, default=None
            index of stop byte.
        bytes_expected : int, default=0
            expected count of bytes.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename="data",
            fmt=fmt,
            order=order,
            stop=stop,
            bytes_expected=bytes_expected,
            default=default,
        )

    @classmethod
    def data_length(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        behaviour: Code = Code.ACTUAL,
        units: Code = Code.BYTES,
        additive: int = 0,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for data length field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        behaviour: Code, default=Code.ACTUAL
            data length field behaviour.
        units: Code, default=Code.BYTES
            data length units.
        additive: int, default=0
            additive value for data length value.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename="data_length",
            fmt=fmt,
            order=order,
            behaviour=behaviour,
            units=units,
            additive=additive,
            default=default,
        )

    @classmethod
    def id_(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for id field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(typename="id", fmt=fmt, order=order, default=default)

    @classmethod
    def operation(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        descriptions: dict[Code, int] | None = None,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for crc field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        descriptions: dict[Code, int] | None, default=None
            operation value descriptions.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        if descriptions is None:
            descriptions = {Code.READ: 0, Code.WRITE: 1}
        return cls(
            typename="operation",
            fmt=fmt,
            order=order,
            descriptions=descriptions,
            default=default,
        )

    @classmethod
    def response(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        descriptions: dict[int, Code] | None = None,
        default_code: Code = Code.UNDEFINED,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for crc field.

        Parameters
        ----------
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        descriptions: dict[Code, int] | None, default=None
            response value descriptions.
        default_code: Code=Code.UNDEFINED
            default code for response value.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        if descriptions is None:
            descriptions = {}
        return cls(
            typename="response",
            fmt=fmt,
            order=order,
            descriptions=descriptions,
            default_code=default_code,
            default=default,
        )

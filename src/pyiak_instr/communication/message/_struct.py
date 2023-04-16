"""Private module of ``pyiak_instr.communication.message`` with field
structs."""
from dataclasses import dataclass, field as _field
from typing import ClassVar, Union

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
        "base": MessageFieldStruct,
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

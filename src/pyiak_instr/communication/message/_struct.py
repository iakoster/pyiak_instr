"""Private module of ``pyiak_instr.communication.message`` with field
structs."""
from __future__ import annotations
from dataclasses import field as _field
from typing import ClassVar, Iterable, Self, Union

from ...core import Code
from ...utilities import BytesEncoder
from ...store import BytesFieldStruct
from ...exceptions import NotAmongTheOptions, ContentError
from ...types.communication import MessageFieldPatternABC
from ...types.store import STRUCT_DATACLASS


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


@STRUCT_DATACLASS
class MessageFieldStruct(BytesFieldStruct):
    """Represents a general field of a Message."""


@STRUCT_DATACLASS
class SingleMessageFieldStruct(MessageFieldStruct):
    """
    Represents a field of a Message with single word.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.words_expected != 1:
            raise ValueError(
                f"{self.__class__.__name__} should expect one word"
            )

    def _modify_values(self) -> None:
        if self.stop is None and self.bytes_expected == 0:
            object.__setattr__(self, "bytes_expected", self.word_bytesize)
        super()._modify_values()


@STRUCT_DATACLASS
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


@STRUCT_DATACLASS
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


@STRUCT_DATACLASS
class CrcMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with crc value.
    """

    poly: int = 0x1021

    init: int = 0

    wo_fields: set[str] = _field(default_factory=set)

    def __post_init__(self) -> None:
        super().__post_init__()
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
class DataMessageFieldStruct(MessageFieldStruct):
    """Represents a field of a Message with data."""


@STRUCT_DATACLASS
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

    # todo: auto encode type
    def calculate(self, data: bytes, fmt: Code) -> int:
        """
        Calculate length of `data`.

        Parameters
        ----------
        data : bytes
            data bytes.
        fmt : Code
            data fmt.

        Returns
        -------
        int
            `data` length.

        Raises
        ------
        ContentError
            if data not integer count of words.
        """
        if self.units is Code.WORDS:
            bytesize = BytesEncoder.get_bytesize(fmt)
            if len(data) % bytesize != 0:
                raise ContentError(self, "non-integer words count in data")
            return len(data) // bytesize

        # units is a BYTES
        return len(data)


@STRUCT_DATACLASS
class IdMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field with a unique identifier of a particular message.
    """


@STRUCT_DATACLASS
class OperationMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with operation (e.g. read).

    Operation codes are needed to compare the operation when receiving
    a message and generally to understand what operation is written in
    the message.

    If the dictionary is None, the standard value will be assigned
    {READ: 0, WRITE: 1}.
    """

    # todo: class with descriptions type
    descs: dict[int, Code] = _field(
        default_factory=lambda: {0: Code.READ, 1: Code.WRITE}
    )
    """matching dictionary value and codes."""

    descs_r: ClassVar[dict[Code, int]]
    """reversed `descriptions`."""

    def encode(
        self, content: Iterable[int | float] | int | float | Code
    ) -> bytes:
        """
        Encode content to bytes.

        Parameters
        ----------
        content : npt.ArrayLike | Code
            content to encoding.

        Returns
        -------
        bytes
            encoded content.

        Raises
        ------
        ContentError
            if `content` is a code which not represented in `descs`.
        """
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

    def _modify_values(self) -> None:
        super()._modify_values()
        object.__setattr__(
            self,
            "descs_r",
            {
                v: k
                for k, v in self.descs.items()  # pylint: disable=no-member
            },
        )


@STRUCT_DATACLASS
class ResponseMessageFieldStruct(SingleMessageFieldStruct):
    """
    Represents a field of a Message with response field.
    """

    descs: dict[int, Code] = _field(default_factory=dict)
    """matching dictionary value and codes."""

    descs_r: ClassVar[dict[Code, int]]
    """reversed `descriptions`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # pylint: disable=no-member
        object.__setattr__(
            self, "descs_r", {v: k for k, v in self.descs.items()}
        )

    def encode(
        self, content: Iterable[int | float] | int | float | Code
    ) -> bytes:
        """
        Encode content to bytes.

        Parameters
        ----------
        content : npt.ArrayLike | Code
            content to encoding.

        Returns
        -------
        bytes
            encoded content.

        Raises
        ------
        ContentError
            if `content` is a code which not represented in `descs`.
        """
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


MessageFieldStructUnionT = Union[  # pylint: disable=invalid-name
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


class MessageFieldPattern(MessageFieldPatternABC[MessageFieldStructUnionT]):
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
        return cls(
            typename="single",
            bytes_expected=cls.get_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

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
        return cls(
            typename="static",
            bytes_expected=cls.get_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

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
            bytes_expected=cls.get_bytesize(fmt),
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
            bytes_expected=cls.get_bytesize(fmt),
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
            bytes_expected=cls.get_bytesize(fmt),
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
        return cls(
            typename="id",
            bytes_expected=cls.get_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

    @classmethod
    def operation(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        descs: dict[Code, int] | None = None,
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
        descs: dict[Code, int] | None, default=None
            operation value descriptions.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        if descs is None:
            descs = {Code.READ: 0, Code.WRITE: 1}
        return cls(
            typename="operation",
            bytes_expected=cls.get_bytesize(fmt),
            fmt=fmt,
            order=order,
            descs=descs,
            default=default,
        )

    @classmethod
    def response(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        descs: dict[int, Code] | None = None,
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
        descs: dict[Code, int] | None, default=None
            response value descriptions.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        if descs is None:
            descs = {}
        return cls(
            typename="response",
            bytes_expected=cls.get_bytesize(fmt),
            fmt=fmt,
            order=order,
            descs=descs,
            default=default,
        )

    @staticmethod
    def get_bytesize(fmt: Code) -> int:
        """
        Get fmt size in bytes.

        Parameters
        ----------
        fmt : Code
            fmt code.

        Returns
        -------
        int
            fmt bytesize.
        """
        return BytesEncoder.get_bytesize(fmt)

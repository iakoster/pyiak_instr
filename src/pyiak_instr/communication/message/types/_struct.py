"""Private module of ``pyiak_instr.types.communication`` with types for
communication module."""
from dataclasses import field as _field
from typing import (
    Callable,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    Union,
)

from ....exceptions import ContentError, NotAmongTheOptions
from ....core import Code
from ....encoders.types import Encoder
from ....store.bin.types import (
    STRUCT_DATACLASS,
    BytesDecodeT,
    BytesEncodeT,
    BytesFieldStructABC,
    BytesStorageStructABC,
)


__all__ = [
    "STRUCT_DATACLASS",
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
    "MessageFieldStructABCUnionT",
    "MessageStructGetParserABC",
    "MessageStructHasParserABC",
    "MessageStructABC",
]


EncoderT: TypeAlias = Encoder[BytesDecodeT, BytesEncodeT, bytes]
FieldStructT = TypeVar("FieldStructT", bound="MessageFieldStructABC")
MessageStructT = TypeVar("MessageStructT", bound="MessageStructABC")


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

    def __post_init__(
        self,
        encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        super().__post_init__(encoder)
        if self.bytes_expected != 0:
            raise ValueError(f"{self.__class__.__name__} can only be dynamic")


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


MessageFieldStructABCUnionT = Union[
    MessageFieldStructABC,
    SingleMessageFieldStructABC,
    StaticMessageFieldStructABC,
    AddressMessageFieldStructABC,
    CrcMessageFieldStructABC,
    DataMessageFieldStructABC,
    DataLengthMessageFieldStructABC,
    IdMessageFieldStructABC,
    OperationMessageFieldStructABC,
    ResponseMessageFieldStructABC,
]


class MessageStructGetParserABC(Generic[MessageStructT, FieldStructT]):
    def __init__(
        self,
        message: MessageStructT,
        codes: dict[Code, str],
    ) -> None:
        self._msg = message
        self._codes = codes

    @property
    def basic(self) -> MessageFieldStructABC:
        return self(Code.BASIC)

    @property
    def single(self) -> SingleMessageFieldStructABC:
        return self(Code.SINGLE)

    @property
    def static(self) -> StaticMessageFieldStructABC:
        return self(Code.STATIC)

    @property
    def address(self) -> AddressMessageFieldStructABC:
        return self(Code.ADDRESS)

    @property
    def crc(self) -> CrcMessageFieldStructABC:
        return self(Code.CRC)

    @property
    def data(self) -> DataMessageFieldStructABC:
        return self(Code.DATA)

    @property
    def data_length(self) -> DataLengthMessageFieldStructABC:
        return self(Code.DATA_LENGTH)

    @property
    def id_(self) -> IdMessageFieldStructABC:
        return self(Code.ID)

    @property
    def operation(self) -> OperationMessageFieldStructABC:
        return self(Code.OPERATION)

    @property
    def response(self) -> ResponseMessageFieldStructABC:
        return self(Code.RESPONSE)

    def __call__(self, code: Code) -> FieldStructT:
        if code not in self._codes:
            raise TypeError(f"field instance with code {code!r} not found")
        return self._msg[self._codes[code]]


class MessageStructHasParserABC(Generic[MessageStructT, FieldStructT]):
    def __init__(
        self,
        message: MessageStructT,
        codes: dict[Code, str],
    ) -> None:
        self._msg = message
        self._codes = codes

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
        return code in self._codes


@STRUCT_DATACLASS
class MessageStructABC(BytesStorageStructABC[FieldStructT]):
    divisible: bool = False
    """shows that the message can be divided by the infinite field."""

    mtu: int = 1500
    """max size of one message part."""

    _field_type_codes: dict[type[FieldStructT], Code] = _field(
        default_factory=dict, init=False
    )  # ClassVar

    _field_types: dict[Code, str] = _field(default_factory=dict, init=False)

    def __post_init__(self, fields: dict[str, FieldStructT]) -> None:
        super().__post_init__(fields)

        field_types = {}
        for struct in self:
            field_class = struct.__class__
            if field_class not in self._field_type_codes:
                raise KeyError(
                    f"{field_class.__name__} not represented in codes"
                )

            field_code = self._field_type_codes[field_class]
            if field_code not in field_types:
                field_types[field_code] = struct.name

        self._setattr("_field_types", field_types)

        if self.divisible:
            if not self.is_dynamic:
                raise TypeError(
                    f"{self.__class__.__name__} can not be divided because "
                    "it does not have a dynamic field"
                )

            min_mtu = (
                self.minimum_size
                + self._f[self.dynamic_field_name].word_bytesize
            )
            if self.mtu < min_mtu:
                raise ValueError(
                    "MTU value does not allow you to split the message if "
                    f"necessary. The minimum MTU is {min_mtu} "
                    f"(got {self.mtu})"
                )

    @property
    def get(self) -> MessageStructGetParserABC[Self, FieldStructT]:
        """
        Returns
        -------
        MessageStructGetParserABC
            message get parser.
        """
        return MessageStructGetParserABC(self, self._field_types)

    @property
    def has(self) -> MessageStructHasParserABC[Self, FieldStructT]:
        """
        Returns
        -------
        MessageHasParserABC
            message has parser.
        """
        return MessageStructHasParserABC(self, self._field_types)

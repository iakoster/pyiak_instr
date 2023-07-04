"""Private module of ``pyiak_instr.communication.message.types``."""
from __future__ import annotations
from dataclasses import field as _field
from typing import (
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    Union,
    cast,
)

from ...exceptions import ContentError, NotAmongTheOptions
from ...core import Code
from ...encoders.bin import BytesEncodeT
from ...store.bin import (
    STRUCT_DATACLASS,
    Field as BinField,
    Struct as BinStruct,
)


__all__ = [
    "STRUCT_DATACLASS",
    "Basic",
    "Static",
    "Address",
    "Crc",
    "Data",
    "DynamicLength",
    "Id",
    "Operation",
    "Response",
    "FieldUnionT",
    "StructGetParser",
    "StructHasParser",
    "Struct",
]


# todo: refactor (join classes to one and use metaclass)
@STRUCT_DATACLASS
class Basic(BinField):
    """
    Represents a base class for base field.
    """

    is_single: ClassVar[bool] = False
    "indicate that only one word expected."

    def _modify_values(self) -> None:
        if self.is_single and self.stop is None and self.bytes_expected == 0:
            object.__setattr__(self, "bytes_expected", self.word_bytesize)
        super()._modify_values()

    def _verify_modified_values(self) -> None:
        super()._verify_modified_values()
        if self.is_single and self.words_expected != 1:
            raise ValueError(
                f"{self.__class__.__name__} should expect one word"
            )


BasicT = TypeVar("BasicT", bound=Basic)


@STRUCT_DATACLASS
class Static(Basic):
    """
    Represents a base class for field with static single word (e.g. preamble).
    """

    is_single = True

    def verify(self, content: bytes, raise_if_false: bool = False) -> Code:
        """
        Verify that `content` is correct for the given field structure.

        Parameters
        ----------
        content : bytes
            content to verifying.
        raise_if_false : bool, default=False
            raise `ContentError` if content not correct.

        Returns
        -------
        Code
            OK - content is correct, other - is not.

        Raises
        ------
        ContentError
            if `raise_if_false` is True and content is not correct.
        """
        code = super().verify(content, raise_if_false=raise_if_false)
        if code is Code.OK and content != self.default:
            if raise_if_false:
                raise ContentError(
                    self,
                    clarification=(
                        f"{Code.INVALID_CONTENT!r} - '{content.hex(' ')}'"
                    ),
                )
            return Code.INVALID_CONTENT
        return Code.OK

    def _verify_init_values(self) -> None:
        super()._verify_init_values()
        if not self.has_default:
            raise ValueError("default value not specified")


@STRUCT_DATACLASS
class Address(Basic):
    """
    Represents base class for field with address.
    """

    is_single = True

    behaviour: Code = Code.DMA

    units: Code = Code.WORDS

    def _verify_init_values(self) -> None:
        if self.behaviour not in {Code.DMA, Code.STRONG}:
            raise NotAmongTheOptions(
                "behaviour", self.behaviour, {Code.DMA, Code.STRONG}
            )
        if self.units not in {Code.BYTES, Code.WORDS}:
            raise NotAmongTheOptions(
                "units", self.units, {Code.BYTES, Code.WORDS}
            )
        super()._verify_init_values()


@STRUCT_DATACLASS
class Crc(Basic):
    """
    Represents base class for field with crc.
    """

    is_single = True

    fill_value: bytes = b"\x00"

    poly: int = 0x1021

    init: int = 0

    wo_fields: set[str] = _field(default_factory=set)

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

    def _verify_modified_values(self) -> None:
        super()._verify_modified_values()
        if self.bytes_expected != 2 or self.poly != 0x1021 or self.init != 0:
            raise NotImplementedError(
                "Crc algorithm not verified for other values"
            )  # todo: implement for any crc


@STRUCT_DATACLASS
class Data(Basic):
    """Represents a field of a Message with data."""

    def _verify_modified_values(self) -> None:
        super()._verify_modified_values()
        if self.bytes_expected != 0:
            raise ValueError(f"{self.__class__.__name__} can only be dynamic")


# todo: rename to DynamicLength
@STRUCT_DATACLASS
class DynamicLength(Basic):
    """
    Represents base class for field with length of dynamic field.
    """

    is_single = True

    fill_value: bytes = b"\x00"

    behaviour: Code = Code.ACTUAL  # todo: logic
    """determines the behavior of determining the content value."""

    units: Code = Code.BYTES
    """data length units. Data can be measured in bytes or words."""

    additive: int = 0
    """additional value to the length of the data."""

    def calculate(self, data: bytes, value_size: int) -> int:
        """
        Calculate field value based on `data`.

        Parameters
        ----------
        data : bytes
            base on which the new value is calculated.
        value_size : int
            size of one single value.

        Returns
        -------
        int
            value based on `data`.

        Raises
        ------
        ContentError
            if data has non-integer count of words.
        """
        if self.units is Code.WORDS:
            if len(data) % value_size != 0:
                raise ContentError(self, "non-integer words count in data")
            dyn_length = len(data) // value_size
        else:  # units is a BYTES
            dyn_length = len(data)

        return dyn_length + self.additive

    def _verify_init_values(self) -> None:
        if self.additive < 0:
            raise ValueError("additive number must be positive integer")
        if self.behaviour not in {Code.ACTUAL, Code.EXPECTED}:
            raise NotAmongTheOptions("behaviour", self.behaviour)
        if self.units not in {Code.BYTES, Code.WORDS}:
            raise NotAmongTheOptions("units", self.units)


@STRUCT_DATACLASS
class Id(Basic):
    """Represents a field with a unique identifier of a particular message."""

    is_single = True


@STRUCT_DATACLASS
class Operation(Basic):
    """
    Represents base class for field with operation.
    """

    is_single = True

    descs: dict[int, Code] = _field(
        default_factory=lambda: {0: Code.READ, 1: Code.WRITE}
    )
    """matching dictionary value and codes."""

    descs_r: dict[Code, int] = _field(init=False)
    """reversed `descriptions`."""

    def encode(
        self, content: BytesEncodeT | Code, verify: bool = False
    ) -> bytes:
        """
        Encode content to bytes.

        There is can encode Code to bytes.

        Parameters
        ----------
        content : BytesEncodeT | Code
            content to encoding.
        verify : bool, default=False
            verify content after encoding.

        Returns
        -------
        bytes
            encoded content.

        Raises
        ------
        ContentError
            if Code not represented.
        """
        if isinstance(content, Code):
            value = self.desc_r(content)
            if value is None:
                raise ContentError(self, f"can't encode {repr(content)}")
            content = value
        return super().encode(content, verify=True)

    def desc(self, value: int) -> Code:
        """
        Get description code.

        Parameters
        ----------
        value : int
            description value.

        Returns
        -------
        Code
            description code. If Code.UNDEFINED - value not found.
        """
        if value not in self.descs:
            return Code.UNDEFINED
        return self.descs[value]

    def desc_r(self, code: Code) -> int | None:
        """
        Get description value.

        Parameters
        ----------
        code : Code
            description code.

        Returns
        -------
        int | None
            description value. If None - Code not found.
        """
        if code not in self.descs_r:
            return None
        return self.descs_r[code]

    def _modify_values(self) -> None:
        super()._modify_values()
        self._setattr(
            "descs_r",
            {
                v: k
                for k, v in self.descs.items()  # pylint: disable=no-member
            },
        )


@STRUCT_DATACLASS
class Response(Basic):
    """
    Represents base class for field with response.
    """

    is_single = True

    descs: dict[int, Code] = _field(default_factory=dict)
    """matching dictionary value and codes."""

    descs_r: dict[Code, int] = _field(init=False)
    """reversed `descriptions`."""

    def encode(
        self, content: BytesEncodeT | Code, verify: bool = False
    ) -> bytes:
        """
        Encode content to bytes.

        There is can encode Code to bytes.

        Parameters
        ----------
        content : BytesEncodeT | Code
            content to encoding.
        verify : bool, default=False
            verify content after encoding.

        Returns
        -------
        bytes
            encoded content.

        Raises
        ------
        ContentError
            if Code not represented.
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
        self._setattr(
            "descs_r",
            {
                v: k
                for k, v in self.descs.items()  # pylint: disable=no-member
            },
        )


FieldUnionT = Union[  # pylint: disable=invalid-name
    Basic,
    Static,
    Address,
    Crc,
    Data,
    DynamicLength,
    Id,
    Operation,
    Response,
]


@STRUCT_DATACLASS
class Struct(BinStruct[BasicT]):
    """
    Represents base class for message structure.
    """

    divisible: bool = False
    """shows that the message can be divided by the infinite field."""

    mtu: int = 1500
    """max size of one message part."""

    _field_type_codes: dict[type[BasicT], Code] = _field(
        default_factory=dict, init=False
    )  # ClassVar

    _field_types: dict[Code, str] = _field(default_factory=dict, init=False)

    def __post_init__(self, fields: dict[str, BasicT]) -> None:
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

        if self.has.dynamic_length and not self.is_dynamic:
            raise TypeError(
                "dynamic length field without dynamic length detected"
            )

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
    def get(self) -> StructGetParser[Self, BasicT]:
        """
        Returns
        -------
        StructGetParser
            message get parser.
        """
        return StructGetParser(self, self._field_types)

    @property
    def has(self) -> StructHasParser[Self, BasicT]:
        """
        Returns
        -------
        MessageHasParserABC
            message has parser.
        """
        return StructHasParser(self, self._field_types)


StructT = TypeVar("StructT", bound=Struct[Any])


class StructGetParser(Generic[StructT, BasicT]):
    """
    Represents parser to getting specified field struct.
    """

    def __init__(
        self,
        message: StructT,
        codes: dict[Code, str],
    ) -> None:
        self._msg = message
        self._codes = codes

    @property
    def basic(self) -> Basic:
        """
        Returns
        -------
        Basic
            first in message basic field struct.
        """
        return cast(Basic, self(Code.BASIC))

    @property
    def static(self) -> Static:
        """
        Returns
        -------
        Static
            first in message static field struct.
        """
        return cast(Static, self(Code.STATIC))

    @property
    def address(self) -> Address:
        """
        Returns
        -------
        Address
            first in message address field struct.
        """
        return cast(Address, self(Code.ADDRESS))

    @property
    def crc(self) -> Crc:
        """
        Returns
        -------
        Crc
            first in message crc field struct.
        """
        return cast(Crc, self(Code.CRC))

    @property
    def data(self) -> Data:
        """
        Returns
        -------
        Data
            first in message data field struct.
        """
        return cast(Data, self(Code.DATA))

    @property
    def dynamic_length(self) -> DynamicLength:
        """
        Returns
        -------
        DynamicLength
            first in message data length field struct.
        """
        return cast(DynamicLength, self(Code.DYNAMIC_LENGTH))

    @property
    def id_(self) -> Id:
        """
        Returns
        -------
        Id
            first in message id field struct.
        """
        return cast(Id, self(Code.ID))

    @property
    def operation(self) -> Operation:
        """
        Returns
        -------
        Operation
            first in message operation field struct.
        """
        return cast(Operation, self(Code.OPERATION))

    @property
    def response(self) -> Response:
        """
        Returns
        -------
        Response
            first in message response field struct.
        """
        return cast(Response, self(Code.RESPONSE))

    def __call__(self, code: Code) -> BasicT:
        if code not in self._codes:
            raise TypeError(f"field instance with code {code!r} not found")
        return cast(BasicT, self._msg[self._codes[code]])


class StructHasParser(Generic[StructT, BasicT]):
    """
    Represents parser to checking that field type exists.
    """

    def __init__(
        self,
        message: StructT,
        codes: dict[Code, str],
    ) -> None:
        self._msg = message
        self._codes = codes

    @property
    def basic(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has basic field.
        """
        return self(Code.BASIC)

    @property
    def static(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has static field.
        """
        return self(Code.STATIC)

    @property
    def address(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has address field.
        """
        return self(Code.ADDRESS)

    @property
    def crc(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has crc field.
        """
        return self(Code.CRC)

    @property
    def data(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has data field.
        """
        return self(Code.DATA)

    @property
    def dynamic_length(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has data length field.
        """
        return self(Code.DYNAMIC_LENGTH)

    @property
    def id_(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has id field.
        """
        return self(Code.ID)

    @property
    def operation(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has operation field.
        """
        return self(Code.OPERATION)

    @property
    def response(self) -> bool:
        """
        Returns
        -------
        bool
            True - message has response field.
        """
        return self(Code.RESPONSE)

    def __call__(self, code: Code) -> bool:
        return code in self._codes

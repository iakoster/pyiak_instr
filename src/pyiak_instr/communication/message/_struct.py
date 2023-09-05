"""Private module of ``pyiak_instr.communication.message.types``."""
# pylint: disable=too-many-lines
from __future__ import annotations
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
from ...codecs import get_bytes_codec
from ...store.bin import (
    Field as BinField,
    Struct as BinStruct,
)


__all__ = [
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
class Basic(BinField):
    """
    Represents a base class for base field.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    """

    is_single: ClassVar[bool] = False
    "indicate that only one word expected."

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> None:
        if self.is_single and stop is None and bytes_expected == 0:
            bytes_expected = get_bytes_codec(fmt).fmt_bytesize

        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

    def _verify_initialized_field(self) -> None:
        super()._verify_initialized_field()
        if self.is_single and self.words_expected != 1:
            raise ValueError(
                f"{self.__class__.__name__} should expect one word"
            )


BasicT = TypeVar("BasicT", bound=Basic)


class Static(Basic):
    """
    Represents a base class for field with static single word (e.g. preamble).

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    """

    is_single = True

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> None:
        if len(default) == 0:
            raise ValueError("'default' value not specified")
        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

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
            code = Code.INVALID_CONTENT

        if code is not Code.OK and raise_if_false:
            raise ContentError(
                self, clarification=f"{code!r} - '{content.hex(' ')}'"
            )
        return code


class Address(Basic):
    """
    Represents base class for field with address.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    behaviour : {DMA, STRONG}, default=DMA
        address change behavior. DMA (Direct Memory Access) enables address
        shifting.
    units : {BYTES, WORDS}, default=WORDS
        address value units.
    """

    is_single = True

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
        behaviour: Code = Code.DMA,
        units: Code = Code.WORDS,
    ) -> None:
        if behaviour not in {Code.DMA, Code.STRONG}:
            raise NotAmongTheOptions(
                "behaviour", behaviour, {Code.DMA, Code.STRONG}
            )
        if units not in {Code.BYTES, Code.WORDS}:
            raise NotAmongTheOptions("units", units, {Code.BYTES, Code.WORDS})

        self._behaviour = behaviour
        self._units = units

        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

    @property
    def behaviour(self) -> Code:
        """
        Returns
        -------
        Code
            address change behavior.
        """
        return self._behaviour

    @property
    def units(self) -> Code:
        """
        Returns
        -------
        Code
            address value units.
        """
        return self._units


class Crc(Basic):
    """
    Represents base class for field with crc.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    poly : int, default=0x1021
        poly value for crc function.
    init : int, default=0
        init value for crc function.
    wo_fields : set[str], default=None
        set of field names whose contents will not be included in the crc
        calculation.
    """

    is_single = True

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"\x00",
        poly: int = 0x1021,
        init: int = 0,
        wo_fields: set[str] = None,
    ) -> None:
        if wo_fields is None:
            wo_fields = set()

        self._poly = poly
        self._init = init
        self._wo_fields = wo_fields

        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

        if self.bytes_expected != 2 or poly != 0x1021 or init != 0:
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

        crc = self._init
        for byte in content:
            crc ^= byte << 8
            for _ in range(8):
                crc <<= 1
                if crc & 0x10000:
                    crc ^= self._poly
                crc &= 0xFFFF
        return crc

    @property
    def poly(self) -> int:
        """
        Returns
        -------
        int
            poly value for crc function.
        """
        return self._poly

    @property
    def init(self) -> int:
        """
        Returns
        -------
        int
            init value for crc function.
        """
        return self._init

    @property
    def wo_fields(self) -> set[str]:  # todo: protect
        """
        Returns
        -------
        set[str]
            set of field names whose contents will not be included in the crc
            calculation.
        """
        return self._wo_fields


class Data(Basic):
    """
    Represents a field of a Message with data.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    """

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> None:
        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

        if self.bytes_expected != 0:
            raise ValueError(f"{self.__class__.__name__} can only be dynamic")


class DynamicLength(Basic):
    """
    Represents base class for field with length of dynamic field.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    behaviour : {ACTUAL, EXPECTED}, default=ACTUAL
        determines the behavior of determining the content value.
    units : {BYTES, WORDS}, default=WORDS
        data length units.
    additive : int, default=0
        additional value to the length of the data.
    """

    is_single = True

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"\x00",
        behaviour: Code = Code.ACTUAL,
        units: Code = Code.BYTES,
        additive: int = 0,
    ) -> None:
        if additive < 0:
            raise ValueError("additive number must be positive integer")
        if behaviour not in {Code.ACTUAL, Code.EXPECTED}:
            raise NotAmongTheOptions("behaviour", behaviour)
        if units not in {Code.BYTES, Code.WORDS}:
            raise NotAmongTheOptions("units", units)

        self._add = additive
        self._behaviour = behaviour
        self._units = units

        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

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
        if self._units is Code.WORDS:
            if len(data) % value_size != 0:
                raise ContentError(self, "non-integer words count in data")
            dyn_length = len(data) // value_size
        else:  # units is a BYTES
            dyn_length = len(data)

        return dyn_length + self._add

    @property
    def behaviour(self) -> Code:
        """
        Returns
        -------
        Code
            determines the behavior of determining the content value.
        """
        return self._behaviour

    @property
    def units(self) -> Code:
        """
        Returns
        -------
        Code
            data length units.
        """
        return self._units

    @property
    def additive(self) -> int:
        """
        Returns
        -------
        int
            additional value to the length of the data.
        """
        return self._add


class Id(Basic):
    """Represents a field with a unique identifier of a particular message."""

    is_single = True


class Operation(Basic):
    """
    Represents base class for field with operation.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    descs : dict[int, Code], default=None
        matching dictionary value and codes. Default value
        is {0: READ, 1: WRITE}.
    """

    is_single = True

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
        descs: dict[int, Code] = None,
    ) -> None:
        if descs is None:
            descs = {0: Code.READ, 1: Code.WRITE}
        self._descs = descs
        self._descs_r = {v: k for k, v in descs.items()}

        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

    def encode(self, content: Any, verify: bool = False) -> bytes:
        """
        Encode content to bytes.

        There is can encode Code to bytes.

        Parameters
        ----------
        content : Any
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
        if value not in self._descs:
            return Code.UNDEFINED
        return self._descs[value]

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
        if code not in self._descs_r:
            return None
        return self._descs_r[code]


class Response(Basic):
    """
    Represents base class for field with response.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    descs : dict[int, Code], default=None
        matching dictionary value and codes. Default value is {}.
    """

    is_single = True

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
        descs: dict[int, Code] = None,
    ) -> None:
        if descs is None:
            descs = {0: Code.READ, 1: Code.WRITE}
        self._descs = descs
        self._descs_r = {v: k for k, v in descs.items()}

        super().__init__(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            order=order,
            default=default,
        )

    def encode(self, content: Any, verify: bool = False) -> bytes:
        """
        Encode content to bytes.

        There is can encode Code to bytes.

        Parameters
        ----------
        content : Any
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
        if value not in self._descs:
            return Code.UNDEFINED
        return self._descs[value]

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
        if code not in self._descs_r:
            return None
        return self._descs_r[code]


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


class Struct(BinStruct[BasicT]):
    """
    Represents base class for message structure.

    Parameters
    ----------
    name : str, default='std'
        name of storage configuration.
    fields : dict[str, FieldT], default=None
        dictionary with fields.
    divisible : bool, default=False
        shows that the message can be divided by the infinite field.
    mtu : int, default=1500
        max size of one message part.
    """

    _f_type_codes: dict[type[BasicT], Code] = {}  # ClassVar

    def __init__(
        self,
        name: str = "std",
        fields: dict[str, BasicT] = None,
        divisible: bool = False,
        mtu: int = 1500,
    ) -> None:
        super().__init__(name=name, fields=fields)

        self._div = divisible
        self._mtu = mtu

        self._f_types: dict[Code, str] = {}
        for struct in self:
            f_class = struct.__class__
            if f_class not in self._f_type_codes:
                raise KeyError(f"{f_class.__name__} not represented in codes")

            f_code = self._f_type_codes[f_class]
            if f_code not in self._f_types:
                self._f_types[f_code] = struct.name

        if self.has.dynamic_length and not self.is_dynamic:
            raise TypeError(
                "dynamic length field without dynamic length detected"
            )

        if self._div:
            if not self.is_dynamic:
                raise TypeError(
                    f"{self.__class__.__name__} can not be divided without "
                    f"dynamic field"
                )

            min_mtu = (
                self.minimum_size
                + self._f[self.dynamic_field_name].fmt_bytesize
            )
            if self._mtu < min_mtu:
                raise ValueError(
                    "MTU value does not allow you to split the message if "
                    f"necessary. The minimum MTU is {min_mtu} "
                    f"(got {self._mtu})"
                )

    @property
    def get(self) -> StructGetParser[Self, BasicT]:
        """
        Returns
        -------
        StructGetParser
            message get parser.
        """
        return StructGetParser(self, self._f_types)

    @property
    def has(self) -> StructHasParser[Self, BasicT]:
        """
        Returns
        -------
        MessageHasParserABC
            message has parser.
        """
        return StructHasParser(self, self._f_types)

    @property
    def divisible(self) -> bool:
        """
        Returns
        -------
        bool
            shows that the message can be divided by the infinite field.
        """
        return self._div

    @property
    def mtu(self) -> int:
        """
        Returns
        -------
        int
            max size of one message part.
        """
        return self._mtu


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

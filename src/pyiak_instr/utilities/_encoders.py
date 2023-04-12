"""Private module of ``pyiak_instr.utilities`` with functions with nums"""
import re
import itertools
import struct
from typing import Any, Callable, Generator, Iterable, Literal

import numpy as np
import numpy.typing as npt

from ..core import Code
from ..exceptions import CodeNotAllowed


__all__ = ["BytesEncoder", "StringEncoder"]


# todo: add string support
# todo: ContentType
class BytesEncoder:
    """
    Represents class for encoding/decoding numbers and arrays to/from bytes.
    """

    ORDERS: dict[Code, str] = {
        Code.BIG_ENDIAN: ">",
        Code.LITTLE_ENDIAN: "<",
    }

    _U_LENGTHS = {
        Code.U8: 1,
        Code.U16: 2,
        Code.U24: 3,
        Code.U32: 4,
        Code.U40: 5,
        Code.U48: 6,
        Code.U56: 7,
        Code.U64: 8,
    }

    _I_LENGTHS = {
        Code.I8: 1,
        Code.I16: 2,
        Code.I24: 3,
        Code.I32: 4,
        Code.I40: 5,
        Code.I48: 6,
        Code.I56: 7,
        Code.I64: 8,
    }

    _F_LENGTHS = {
        Code.F16: 2,
        Code.F32: 4,
        Code.F64: 6,
    }

    _F_DTYPES = {
        Code.F16: "e",
        Code.F32: "f",
        Code.F64: "d",
    }

    ALLOWED_CODES = set(_U_LENGTHS)
    """types of values for encoding"""

    ALLOWED_CODES.update(_I_LENGTHS)
    ALLOWED_CODES.update(_F_LENGTHS)

    @classmethod
    def decode(
        cls,
        content: bytes,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
    ) -> npt.NDArray[
        np.int_ | np.float_
    ]:  # todo: np.ndarray[int | float, Any]?
        """
        Decode bytes content to array.

        Parameters
        ----------
        content : bytes
            content to decoding.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            bytes order.

        Returns
        -------
        numpy.typing.NDArray
            decoded values.

        Raises
        ------
        CodeNotAllowed
            if `fmt` or `order` not in list of existed formats.
        """
        cls.check_fmt_order(fmt, order)

        if fmt in cls._U_LENGTHS:
            return np.fromiter(
                cls._decode_int(content, fmt, order, False), int
            )

        if fmt in cls._I_LENGTHS:
            return np.fromiter(
                cls._decode_int(content, fmt, order, True), int
            )

        return np.frombuffer(content, dtype=cls._get_dtype(fmt, order))

    @classmethod
    def decode_value(
        cls,
        content: bytes,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
    ) -> np.int_ | np.float_:
        """
        Decode one value from content.

        Parameters
        ----------
        content : bytes
            value content.
        fmt : Code, default=Code.U8
            value format.
        order : Code, order=Code.BIG_ENDIAN
            content order.

        Returns
        -------
        numpy.number
            decoded value.

        Raises
        ------
        ValueError
            if content is not a one value.
        """
        value = cls.decode(content, fmt=fmt, order=order)
        if len(value) != 1:
            raise ValueError("content must be specified by one value")
        return value[0]  # type: ignore[no-any-return]

    @classmethod
    def _decode_int(
        cls, content: bytes, fmt: Code, order: Code, signed: bool
    ) -> Iterable[int]:
        """
        Returns a generator where each element is a decoded integer.

        Parameters
        ----------
        content : bytes
            content for decoding.
        fmt : Code
            value format code.
        order : Code
            order code.
        signed : bool
            indicator indicating whether each value is unsigned or not.

        Yields
        ------
        int
            decoded single value.
        """
        value_length = (
            cls._U_LENGTHS if fmt in cls._U_LENGTHS else cls._I_LENGTHS
        )[fmt]

        byteorder: Literal["big", "little"]  # for typing
        if order is Code.BIG_ENDIAN:
            byteorder = "big"
        else:
            byteorder = "little"

        for val in cls._iterate_by_bytes(content, value_length):
            yield int.from_bytes(val, byteorder, signed=signed)

    @classmethod
    def _iterate_by_bytes(
        cls, content: bytes, value_length: int
    ) -> Iterable[bytes]:
        """
        Returns a generator where each value is a bytes slice with a single
        value.

        Parameters
        ----------
        content : bytes
            content.
        value_length : int
            length of single value in bytes.

        Yields
        ------
        bytes
            value bytes slice.
        """
        for i_val in range(0, len(content) // value_length):
            yield content[i_val * value_length : (i_val + 1) * value_length]

    @classmethod
    def encode(
        cls,
        content: Iterable[int | float] | int | float,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
    ) -> bytes:
        """
        Encode values to bytes.

        Parameters
        ----------
        content : numpy.typing.ArrayLike
            values to encoding.
        fmt : Code, default=Code.U8
            format code.
        order : Code, default=Code.BIG_ENDIAN
            order code.

        Returns
        -------
        bytes
            encoded values.
        """
        cls.check_fmt_order(fmt, order)

        if fmt in cls._U_LENGTHS:
            return b"".join(cls._encode_int(content, fmt, order, False))

        if fmt in cls._I_LENGTHS:
            return b"".join(cls._encode_int(content, fmt, order, True))

        return np.array(content, dtype=cls._get_dtype(fmt, order)).tobytes()

    @classmethod
    def _encode_int(
        cls,
        content: Iterable[int | float] | int | float,
        fmt: Code,
        order: Code,
        signed: bool,
    ) -> Iterable[bytes]:
        """
        Returns a generator where each value is an encoded integer.

        Parameters
        ----------
        content : ArrayLike
            values for encoding.
        fmt : Code
            value format code.
        order : Code
            order code.
        signed : bool
            indicator indicating whether each value is unsigned or not.

        Yields
        ------
        int
            encoded single value.
        """
        value_length = (
            cls._U_LENGTHS if fmt in cls._U_LENGTHS else cls._I_LENGTHS
        )[fmt]

        byteorder: Literal["big", "little"]  # for typing
        if order is Code.BIG_ENDIAN:
            byteorder = "big"
        else:
            byteorder = "little"

        if not isinstance(content, Iterable):
            content = [content]

        for value in content:
            if not isinstance(value, int):
                value = int(value)
            yield value.to_bytes(value_length, byteorder, signed=signed)

    @classmethod
    def _get_dtype(cls, fmt: Code, order: Code) -> str:
        """
        Check the correctness of `fmt` and `order` and get format string.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code
            order code.

        Returns
        -------
        str
            format string.

        Raises
        ------
        CodeNotAllowed
            if `fmt` or `order` not in list of existed formats.
        """
        assert fmt in cls._F_DTYPES

        dtype = cls._F_DTYPES[fmt]
        if struct.calcsize(dtype) > 1:
            dtype = cls.ORDERS[order] + dtype
        return dtype

    @classmethod
    def check_fmt_order(cls, fmt: Code, order: Code) -> None:
        """
        Check that `fmt` and `order` codes is allowed.

        Parameters
        ----------
        fmt : Code
            value format code.
        order : Code
            order code.

        Raises
        ------
        CodeNotAllowed
            if `fmt` or `order` not in list of existed formats.
        """
        if fmt not in cls.ALLOWED_CODES:
            raise CodeNotAllowed(fmt)
        if order not in cls.ORDERS:
            raise CodeNotAllowed(order)

    @classmethod
    def get_bytesize(cls, fmt: Code) -> int:
        """
        Get format size in bytes.

        Parameters
        ----------
        fmt : Code
            format code.

        Returns
        -------
        int
            bytesize.

        Raises
        ------
        CodeNotAllowed
            if `fmt` not in list of allowed codes.
        """
        if fmt in cls._U_LENGTHS:
            return cls._U_LENGTHS[fmt]
        if fmt in cls._I_LENGTHS:
            return cls._I_LENGTHS[fmt]
        if fmt in cls._F_LENGTHS:
            return cls._F_LENGTHS[fmt]
        raise CodeNotAllowed(fmt)


# todo: parameters (e.g. \npa[shape=\tpl(2,1),dtype=uint8](1,2))
class StringEncoder:
    """
    Represent class for encoding/decoding python values to/from string.
    """

    DELIMITER = ","
    "Delimiter between values"

    FLOAT = re.compile(r"^-?\d+\.\d+([eE][+-]\d+)?$")
    "Float pattern"

    SOH = "\\"
    "Start of header"

    SOD = "("
    "Start of data"

    EOD = ")"
    "End of data"

    HEADERS = {
        "bts": Code.BYTES,
        "cod": Code.CODE,
        "dct": Code.DICT,
        "lst": Code.LIST,
        "set": Code.SET,
        "str": Code.STRING,
        "tpl": Code.TUPLE,
        "npa": Code.NUMPY_ARRAY,
    }
    "Dictionary with headers of complex types"

    _HEADERS_R = {v: k for k, v in HEADERS.items()}

    _COMPLEX_TYPES = {
        bytes: Code.BYTES,
        dict: Code.DICT,
        list: Code.LIST,
        set: Code.SET,
        tuple: Code.TUPLE,
        np.ndarray: Code.NUMPY_ARRAY,
    }
    "Types where encoded string must be complex"

    COMPLEX_TYPES: set[type] = {*_COMPLEX_TYPES}
    "Set of types that require coding"

    _DECODERS: dict[Code, Callable[[Any], Any]] = {
        Code.NONE: lambda x: None,
        Code.BOOL: lambda x: x == "True",
        Code.BYTES: bytes,
        Code.CODE: Code,
        Code.DICT: lambda x: {v: next(x) for v in x},
        Code.FLOAT: float,
        Code.INT: int,
        Code.LIST: list,
        Code.SET: set,
        Code.STRING: lambda x: x,
        Code.TUPLE: tuple,
        Code.NUMPY_ARRAY: lambda x: np.array(tuple(x)),
    }
    "Dictionary of decoders"

    @classmethod
    def decode(cls, string: str) -> Any:
        """
        Decode value from `string`.

        If conversion is not possible, it returns the string as is.

        Parameters
        ----------
        string: str
            value encoded in the string.

        Returns
        -------
        Any
            decoded value.
        """
        if cls._is_compound_string(string):
            code, value = cls._read(string)
            if code is Code.STRING:
                return value
            if code is Code.CODE:
                return Code(int(value))
            return cls._DECODERS[code](
                map(cls._decode_value, cls._iter(value))
            )
        return cls._decode_value(string)

    @classmethod
    def encode(cls, value: Any) -> str:
        """
        Encode value to the string.

        Parameters
        ----------
        value: Any
            value for encoding.

        Returns
        -------
        str
            encoded value.
        """
        if isinstance(value, str):
            # todo: optimize - check only SOH and numbers
            if value == cls._decode_value(value):
                return value
            return cls._decorate(Code.STRING, value)

        if type(value) not in cls._COMPLEX_TYPES:
            return cls._encode_value(value)

        code = cls._COMPLEX_TYPES[type(value)]
        if code is Code.DICT:
            value = itertools.chain.from_iterable(value.items())
        return cls._decorate(
            code, cls.DELIMITER.join(map(cls._encode_value, value))
        )

    @classmethod
    def _decode_value(cls, value: str) -> Any:
        """
        Convert the value from a string (if possible) or return the value
        as is.

        Parameters
        ----------
        value: str
            single value for decoding.

        Returns
        -------
        Any
            decoded value.
        """
        if cls._is_compound_string(value):
            return cls.decode(value)
        return cls._DECODERS[cls._determine_type(value)](value)

    @classmethod
    def _decorate(cls, code: Code, string: str) -> str:
        """
        Add `SOH`, header, `SOT` and `EOT` to the string.

        Parameters
        ----------
        code: Code
            code for header.
        string: str
            string for wrapping.

        Returns
        -------
        str
            decorated string.
        """
        return cls.SOH + cls._HEADERS_R[code] + cls.SOD + string + cls.EOD

    @classmethod
    def _determine_type(cls, string: str) -> Code:
        """
        Get type code.

        Parameters
        ----------
        string : str
            not compound string.

        Returns
        -------
        Code
            type code.
        """
        if len(string) == 0:
            return Code.STRING

        if string[0] == "-" and string[1:].isdigit() or string.isdigit():
            return Code.INT

        if cls.FLOAT.match(string) is not None:
            return Code.FLOAT

        if string in ("True", "False"):
            return Code.BOOL

        if string == "None":
            return Code.NONE

        return Code.STRING

    @classmethod
    def _encode_value(cls, value: Any) -> str:
        """
        Convert value to string.

        Parameters
        ----------
        value: Any
            single value for encoding.

        Returns
        -------
        str
            encoded value.
        """
        if type(value) in cls.COMPLEX_TYPES or isinstance(value, str):
            return cls.encode(value)
        if isinstance(value, Code):
            return cls._decorate(Code.CODE, str(value))
        return str(value)

    @classmethod
    def _get_data_border(cls, string: str) -> tuple[int, int]:
        """
        find `SOD` and `EOD` for `SOH` at the beginning of the string.

        Parameters
        ----------
        string: str
            string.

        Returns
        -------
        tuple[int, int]
            `SOD` and `EOD` positions.

        Raises
        ------
        ValueError
            if `string` does not have `SOD`;
            if `SOD` in `string` does not close.
        """
        sod_pos, opened_sod = string.find(cls.SOD), 1
        if sod_pos < 0:
            raise ValueError("string does not have SOD")

        for i_char in range(sod_pos + 1, len(string)):
            char = string[i_char]

            if char == cls.SOD:
                opened_sod += 1
            elif char == cls.EOD:
                opened_sod -= 1
                if not opened_sod:
                    return sod_pos, i_char

        raise ValueError("SOD not closed in %r" % string)

    @classmethod
    def _is_compound_string(cls, string: str) -> bool:
        """
        Check that string can be converted to value.

        Parameters
        ----------
        string: str
            string for checking.

        Returns
        -------
        bool
            True - if `string` is compound, otherwise False.
        """
        return (
            len(string) >= 6
            and string[0] == cls.SOH
            and cls.SOD in string
            and cls.EOD in string
            and string[1:4] in cls.HEADERS
        )

    @classmethod
    def _iter(cls, string: str) -> Generator[str, None, None]:
        """
        Iterate string by values separated by a `DELIMITER`.

        Parameters
        ----------
        string: str
            string for iterating.

        Yields
        ------
        str
            single raw value.
        """
        length, i, raw = len(string), 0, ""
        while i < length:
            char = string[i]

            if char == cls.SOH:
                assert len(raw) == 0, "raw value not empty"
                _, eod = cls._get_data_border(string[i:])
                yield string[i : i + eod + 1]
                i += eod + 1

            elif char == cls.DELIMITER:
                yield raw
                raw = ""

            else:
                raw += char
            i += 1

        if len(raw) != 0 and len(string) != 0:
            yield raw

    @classmethod
    def _read(cls, string: str) -> tuple[Code, str]:
        """
        Read header and get Code with clear data.

        Parameters
        ----------
        string: str
            raw string

        Returns
        -------
        tuple[Code, str]
            header code and clear data.
        """
        head, data = cls._split(string)
        return cls.HEADERS[head], data

    @classmethod
    def _split(cls, string: str) -> tuple[str, str]:
        """
        Split string with `SOH`, `SOD` and `EOD` to head and data.

        Parameters
        ----------
        string : str
            string for splitting

        Returns
        -------
        tuple[str, str]
            parts of the `string`.
        """
        sod, eod = cls._get_data_border(string)
        return string[sod - 3 : sod], string[sod + 1 : eod]

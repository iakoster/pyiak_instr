"""Private module of ``pyiak_instr.encoders``"""
import re
import itertools
from typing import Any, Callable, Generator, Iterable, Literal

import numpy as np
import numpy.typing as npt

from ..core import Code
from ..exceptions import CodeNotAllowed
from .types import Encoder


__all__ = ["BytesDecodeT", "BytesEncodeT", "BytesEncoder", "StringEncoder"]


BytesDecodeT = npt.NDArray[np.int_ | np.float_]
BytesEncodeT = (
    int | float | bytes | list[int | float] | npt.NDArray[np.int_ | np.float_]
)


# todo: add string support
# todo: ContentType
class BytesEncoder(Encoder[BytesDecodeT, BytesEncodeT, bytes]):
    """
    Represents class for encoding/decoding numbers and arrays to/from bytes.

    Parameters
    ----------
    fmt : Code, default=Code.U8
            value format.
    order : Code, default=Code.BIG_ENDIAN
        bytes order.
    """

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
        Code.F64: 8,
    }

    _F_DTYPES = {
        Code.F16: "e",
        Code.F32: "f",
        Code.F64: "d",
    }

    _F_ORDERS = {
        Code.BIG_ENDIAN: ">",
        Code.LITTLE_ENDIAN: "<",
    }

    ALLOWED_CODES = set(_U_LENGTHS)
    """types of values for encoding"""

    ALLOWED_CODES.update(_I_LENGTHS)
    ALLOWED_CODES.update(_F_LENGTHS)

    def __init__(
        self, fmt: Code = Code.U8, order: Code = Code.BIG_ENDIAN
    ) -> None:
        self.verify_fmt_order(fmt, order)
        self._fmt, self._order = fmt, order
        self._decode_func, self._encode_func = self._get_funcs(fmt, order)

    # todo: return np.ndarray[int | float, Any]?
    def decode(self, value: bytes) -> BytesDecodeT:
        """
        Decode bytes content to array.

        Parameters
        ----------
        value : bytes
            content to decoding.

        Returns
        -------
        DecodeT
            decoded values.

        Raises
        ------
        CodeNotAllowed
            if `fmt` or `order` not in list of existed formats.
        """
        return self._decode_func(value)

    @staticmethod
    def _decode_int(
        value: bytes,
        bytesize: int,
        byteorder: Literal["little", "big"],
        signed: bool,
    ) -> BytesDecodeT:
        """
        Decode bytes to integer array.

        Parameters
        ----------
        value : bytes
            value to decoding.
        bytesize : int
            value size in bytes.
        byteorder : Literal["little", "big"]
            order of bytes (little or big endian).
        signed : bool
            True - value is signed integer.

        Returns
        -------
        BytesDecodeT
            decoded value.
        """
        encoded = np.empty(len(value) // bytesize, np.int_)
        for i in range(encoded.shape[0]):
            val = value[i * bytesize : (i + 1) * bytesize]
            encoded[i] = int.from_bytes(val, byteorder, signed=signed)
        return encoded

    @staticmethod
    def _decode_float(value: bytes, dtype: str) -> BytesDecodeT:
        """
        Decode bytes to float array.

        Parameters
        ----------
        value : bytes
            value to decoding.
        dtype : str
            value type.

        Returns
        -------
        BytesDecodeT
            decoded value.
        """
        return np.frombuffer(value, dtype=dtype)

    def encode(self, value: BytesEncodeT) -> bytes:
        """
        Encode values to bytes.

        If value is instance of bytes type - return value as is.

        Parameters
        ----------
        value : EncodeT
            values to encoding.

        Returns
        -------
        bytes
            encoded values.
        """
        if isinstance(value, bytes):
            return value
        return self._encode_func(value)

    @staticmethod
    def _encode_int(
        value: BytesEncodeT,
        bytesize: int,
        byteorder: Literal["little", "big"],
        signed: bool,
    ) -> bytes:
        """
        Encode integer to bytes.

        Parameters
        ----------
        value : BytesEncodeT
            value to encoding.
        bytesize : int
            value size in bytes.
        byteorder : Literal["little", "big"]
            order of bytes (little or big endian).
        signed : bool
            True - value is signed integer.

        Returns
        -------
        bytes
            encoded value.
        """
        encoded = b""
        if isinstance(value, Iterable):
            for val in value:
                encoded += int(val).to_bytes(
                    bytesize, byteorder, signed=signed
                )
        else:
            encoded += int(value).to_bytes(bytesize, byteorder, signed=signed)
        return encoded

    @staticmethod
    def _encode_float(value: BytesEncodeT, dtype: str) -> bytes:
        """
        Encode float to bytes.

        Parameters
        ----------
        value : BytesEncodeT
            value to encoding.
        dtype : str
            value type.

        Returns
        -------
        bytes
            encoded value.
        """
        return np.array(value, dtype=dtype).tobytes()

    def _get_funcs(
        self, fmt: Code, order: Code
    ) -> tuple[
        Callable[[bytes], BytesDecodeT], Callable[[BytesEncodeT], bytes]
    ]:
        """
        Get decode and encode functions.

        Parameters
        ----------
        fmt : Code
            fmt code.
        order : Code
            order code.

        Returns
        -------
        tuple[Callable, Callable]
            * decode function;
            * encode function.
        """
        if fmt in self._F_LENGTHS:
            dtype = (
                ">" if order is Code.BIG_ENDIAN else "<"
            ) + self._F_DTYPES[fmt]
            return (
                lambda v: self._decode_float(v, dtype),
                lambda v: self._encode_float(v, dtype),
            )

        # fmt in _U_LENGTH or _I_LENGTH
        byteorder: Literal["little", "big"]
        if order is Code.BIG_ENDIAN:
            byteorder = "big"
        else:
            byteorder = "little"

        signed = fmt in self._I_LENGTHS
        bytesize = self._I_LENGTHS[fmt] if signed else self._U_LENGTHS[fmt]
        return (
            lambda v: self._decode_int(v, bytesize, byteorder, signed),
            lambda v: self._encode_int(v, bytesize, byteorder, signed),
        )

    def verify_fmt_order(self, fmt: Code, order: Code) -> None:
        """
        Check that `fmt` and `order` codes is allowed.

        Parameters
        ----------
        fmt : Code
            value format code.
        order : Code
            order code.
        """
        self.verify_fmt(fmt)
        self.verify_order(order)

    def verify_fmt(self, fmt: Code) -> None:
        """
        Check that `fmt` codes is allowed.

        Parameters
        ----------
        fmt : Code
            value format code.

        Raises
        ------
        CodeNotAllowed
            if `fmt` not allowed.
        """
        if fmt not in self.ALLOWED_CODES:
            raise CodeNotAllowed(fmt)

    @staticmethod
    def verify_order(order: Code) -> None:
        """
        Check that `order` codes is allowed.

        Parameters
        ----------
        order : Code
            order code.

        Raises
        ------
        CodeNotAllowed
            if `order` not allowed.
        """
        if order not in {Code.BIG_ENDIAN, Code.LITTLE_ENDIAN}:
            raise CodeNotAllowed(order)

    @property
    def value_size(self) -> int:
        """
        Returns
        -------
        int
            single value size.

        Raises
        ------
        AssertionError
            if in some reason fmt not in allowed codes.
        """
        if self._fmt in self._U_LENGTHS:
            return self._U_LENGTHS[self._fmt]
        if self._fmt in self._I_LENGTHS:
            return self._I_LENGTHS[self._fmt]
        if self._fmt in self._F_LENGTHS:
            return self._F_LENGTHS[self._fmt]
        raise AssertionError(f"invalid value format: {self._fmt!r}")


# todo: parameters (e.g. \npa[shape=\tpl(2,1),dtype=uint8](1,2))
class StringEncoder(Encoder[Any, Any, str]):
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

    def decode(self, value: str) -> Any:
        """
        Decode value from `string`.

        If conversion is not possible, it returns the string as is.

        Parameters
        ----------
        value: str
            value encoded in the string.

        Returns
        -------
        Any
            decoded value.
        """
        if self._is_compound_string(value):
            code, value = self._read(value)
            if code is Code.STRING:
                return value
            if code is Code.CODE:
                return Code(int(value))
            return self._DECODERS[code](
                map(self._decode_value, self._iter(value))
            )
        return self._decode_value(value)

    def encode(self, value: Any) -> str:
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
            if value == self._decode_value(value):
                return value
            return self._decorate(Code.STRING, value)

        if type(value) not in self._COMPLEX_TYPES:
            return self._encode_value(value)

        code = self._COMPLEX_TYPES[type(value)]
        if code is Code.DICT:
            value = itertools.chain.from_iterable(value.items())
        return self._decorate(
            code, self.DELIMITER.join(map(self._encode_value, value))
        )

    def _decode_value(self, value: str) -> Any:
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
        if self._is_compound_string(value):
            return self.decode(value)
        return self._DECODERS[self._determine_type(value)](value)

    def _decorate(self, code: Code, string: str) -> str:
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
        return self.SOH + self._HEADERS_R[code] + self.SOD + string + self.EOD

    def _determine_type(self, string: str) -> Code:
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

        if self.FLOAT.match(string) is not None:
            return Code.FLOAT

        if string in ("True", "False"):
            return Code.BOOL

        if string == "None":
            return Code.NONE

        return Code.STRING

    def _encode_value(self, value: Any) -> str:
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
        if type(value) in self.COMPLEX_TYPES or isinstance(value, str):
            return self.encode(value)
        if isinstance(value, Code):
            return self._decorate(Code.CODE, str(value))
        return str(value)

    def _get_data_border(self, string: str) -> tuple[int, int]:
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
        sod_pos, opened_sod = string.find(self.SOD), 1
        if sod_pos < 0:
            raise ValueError("string does not have SOD")

        for i_char in range(sod_pos + 1, len(string)):
            char = string[i_char]

            if char == self.SOD:
                opened_sod += 1
            elif char == self.EOD:
                opened_sod -= 1
                if not opened_sod:
                    return sod_pos, i_char

        raise ValueError(f"SOD not closed in '{string}'")

    def _is_compound_string(self, string: str) -> bool:
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
            and string[0] == self.SOH
            and self.SOD in string
            and self.EOD in string
            and string[1:4] in self.HEADERS
        )

    def _iter(self, string: str) -> Generator[str, None, None]:
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

            if char == self.SOH:
                assert len(raw) == 0, "raw value not empty"
                _, eod = self._get_data_border(string[i:])
                yield string[i : i + eod + 1]
                i += eod + 1

            elif char == self.DELIMITER:
                yield raw
                raw = ""

            else:
                raw += char
            i += 1

        if len(raw) != 0 and len(string) != 0:
            yield raw

    def _read(self, string: str) -> tuple[Code, str]:
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
        head, data = self._split(string)
        return self.HEADERS[head], data

    def _split(self, string: str) -> tuple[str, str]:
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
        sod, eod = self._get_data_border(string)
        return string[sod - 3 : sod], string[sod + 1 : eod]

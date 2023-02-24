import re
import itertools
from typing import Any, Iterable, Generator

import numpy as np

from src.pyiak_instr.core import Code


__all__ = ["StringEncoder"]


# todo: parameters (e.g. \npa[shape=\tpl(2,1),dtype=uint8](1,2))
class StringEncoder(object):

    SUPPORTED_ITERABLES = bytes | dict | list | set | tuple | np.ndarray
    "Union type of supported iterables (expect string)"

    DELIMITER = ","
    "Delimiter between values"

    FLOAT = re.compile("^-?\d\.\d+([eE][+-]\d+)?$")
    "Float pattern"

    SOH = "\\"
    "Start of header"

    SOD = "("
    "Start of data"

    EOD = ")"
    "End of data"

    HEADERS = {
        "bts": Code.BYTES,
        "dct": Code.DICT,
        "lst": Code.LIST,
        "npa": Code.NUMPY_ARRAY,
        "set": Code.SET,
        "str": Code.STRING,
        "tpl": Code.TUPLE,
    }
    "Dictionary with headers of various types"

    _HEADERS_R = {v: k for k, v in HEADERS.items()}

    _HARD_TYPES = {
        bytes: Code.BYTES,
        dict: Code.DICT,
        list: Code.LIST,
        np.ndarray: Code.NUMPY_ARRAY,
        set: Code.SET,
        tuple: Code.TUPLE,
    }
    "Types where encoded string must be compound"

    _DECODERS = {
        Code.NONE: lambda x: None,
        Code.BOOL: lambda x: x == "True",
        Code.BYTES: bytes,
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
            if value == cls._decode_value(value):
                return value
            return cls._decorate(Code.STRING, value)

        if type(value) not in cls._HARD_TYPES:
            return cls._encode_value(value)

        code = cls._HARD_TYPES[type(value)]
        if code is Code.DICT:
            value = itertools.chain.from_iterable(value.items())
        return cls._decorate(
            code, cls.DELIMITER.join(map(cls._encode_value, value))
        )

    @classmethod
    def _decode_value(cls, value: str) -> Any:
        """
        Convert the value from a string (if possible)
        or return the value as is.
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
        if not len(string):
            return Code.STRING

        elif string[0] == "-" and string[1:].isdigit() or string.isdigit():
            return Code.INT

        elif cls.FLOAT.match(string) is not None:
            return Code.FLOAT

        elif string in ("True", "False"):
            return Code.BOOL

        elif string == "None":
            return Code.NONE

        return Code.STRING

    @classmethod
    def _encode_value(cls, value: Any) -> str:
        """Convert value to string."""
        if isinstance(value, cls.SUPPORTED_ITERABLES | str):
            return cls.encode(value)
        elif isinstance(value, Code):
            value = value.value
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
        """Check that string can be converted to value."""
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
                assert not len(raw), "raw value not empty"
                _, eod = cls._get_data_border(string[i:])
                yield string[i : i + eod + 1]
                i += eod + 1

            elif char == cls.DELIMITER:
                yield raw
                raw = ""

            else:
                raw += char
            i += 1

        if len(raw) and len(string):
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

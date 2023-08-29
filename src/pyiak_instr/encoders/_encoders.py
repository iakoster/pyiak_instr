"""Private module of ``pyiak_instr.encoders``"""
import re
import itertools
from typing import Any, Callable, Generator

import numpy as np

from ..core import Code
from .types import Decoder, Encoder


__all__ = ["StringEncoder"]


# todo: parameters (e.g. \npa[shape=\tpl(2,1),dtype=uint8](1,2))
class StringEncoder(Decoder[Any, str], Encoder[Any, str]):
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

    def decode(self, data: str) -> Any:
        """
        Decode value from `string`.

        If conversion is not possible, it returns the string as is.

        Parameters
        ----------
        data: str
            value encoded in the string.

        Returns
        -------
        Any
            decoded value.
        """
        if self._is_compound_string(data):
            code, data = self._read(data)
            if code is Code.STRING:
                return data
            if code is Code.CODE:
                return Code(int(data))
            return self._DECODERS[code](
                map(self._decode_value, self._iter(data))
            )
        return self._decode_value(data)

    def encode(self, data: Any) -> str:
        """
        Encode value to the string.

        Parameters
        ----------
        data: Any
            value for encoding.

        Returns
        -------
        str
            encoded value.
        """
        if isinstance(data, str):
            # todo: optimize - check only SOH and numbers
            if data == self._decode_value(data):
                return data
            return self._decorate(Code.STRING, data)

        if type(data) not in self._COMPLEX_TYPES:
            return self._encode_value(data)

        code = self._COMPLEX_TYPES[type(data)]
        if code is Code.DICT:
            data = itertools.chain.from_iterable(data.items())
        return self._decorate(
            code, self.DELIMITER.join(map(self._encode_value, data))
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

import re
import itertools
from typing import Any, Generator

import numpy as np

from ..core import Code


__all__ = [
    "split_complex_dict",
    "StringEncoder",
]


def split_complex_dict(
        complex_dict: dict[str, Any],
        sep: str = "__",
        without_sep: str = "raise"
) -> (
        dict[str, dict[str, Any]]
        | tuple[dict[str, dict[str, Any]], dict[str, Any]]
):
    """
    Split dictionary to nested dictionaries (subdictionaries) by `sep`.

    Parameters
    ----------
    complex_dict: dict[str, Any]
        dictionary for splitting.
    sep: str, default='__'
        separator of nested dictionaries.
    without_sep: str, default='raise'
        behavior if key without sep is detected: 'raise' - raise error,
        'other' - put that keys to other dictionary.

    Returns
    -------
    result: dict[str, dict[str, Any]]
        nested dictionaries separated by `sep`.
    dict_without_sep: dict[str, Any], optional
        dictionary with keys without `sep`. It will be empty if there is no
        keys without `sep` and `without_sep`='other'.

    Raises
    ------
    ValueError
        if keyword argument `without_sep` not in {'raise', 'other'}.
    KeyError
        if `without_sep' is equal to 'raise' and key does not have 'sep'.
    """
    if without_sep not in {"raise", "other"}:
        raise ValueError(
            "invalid attribute 'without_sep': %r not in {'raise', 'other'}"
            % without_sep
        )

    result, wo_sep_dict = {}, {}
    for key, value in complex_dict.items():
        if without_sep == "raise" and sep not in key:
            raise KeyError("key %r does not have separator %r" % (key, sep))
        elif sep not in key:
            wo_sep_dict[key] = value
            continue

        sub_keys = key.split(sep)
        sub_dict = result

        for i_sub_key, sub_key in enumerate(sub_keys):
            if i_sub_key == len(sub_keys) - 1:
                break

            if sub_key not in sub_dict:
                sub_dict[sub_key] = {}
            sub_dict = sub_dict[sub_key]

        sub_dict[sub_keys[-1]] = value

    if without_sep == "other":
        return result, wo_sep_dict
    return result


# todo: parameters (e.g. \npa[shape=\tpl(2,1),dtype=uint8](1,2))
class StringEncoder(object):

    SINGLE_TYPES = int | float | str | bool | None
    ITERS = dict | list | np.ndarray | set | tuple | bytes

    DELIMITER = ","

    INT = re.compile("^-?\d+$")
    FLOAT = re.compile("^-?\d\.\d+([eE][+-]\d+)?$")

    SOH = "\\"          # start of header
    SOT = "("          # start of text
    EOT = ")"           # end of value

    HEADERS = {
        "bts": Code.BYTES,
        "dct": Code.DICT,
        "lst": Code.LIST,
        "npa": Code.NUMPY_ARRAY,
        "set": Code.SET,
        "str": Code.STRING,
        "tpl": Code.TUPLE,
    }
    _HEADERS_R = {v: k for k, v in HEADERS.items()}
    _CONVERTERS = {
        Code.BYTES: lambda s: bytes(v for v in s),
        Code.DICT: lambda v: {i: next(v) for i in v},
        Code.LIST: list,
        Code.NUMPY_ARRAY: lambda v: np.array(tuple(v)),
        Code.SET: set,
        Code.TUPLE: tuple,
    }
    _HARD_TYPES = {
        bytes: Code.BYTES,
        dict: Code.DICT,
        list: Code.LIST,
        np.ndarray: Code.NUMPY_ARRAY,
        set: Code.SET,
        tuple: Code.TUPLE,
    }

    @classmethod
    def decode(cls, value: str) -> Any:
        """
        Decode value from string.

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
        if cls._soh_exists(value):
            code, value = cls._read_header(value)
            if code is Code.STRING:
                return value
            return cls._CONVERTERS[code](
                map(cls._to_value, cls._iter_string(value))
            )
        return cls._to_value(value)

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
            if value == cls._to_value(value):
                return value
            return cls._decorate(Code.STRING, value)

        if type(value) not in cls._HARD_TYPES:
            return cls._to_string(value)

        code = cls._HARD_TYPES[type(value)]
        if isinstance(value, dict):
            value = itertools.chain.from_iterable(value.items())
        return cls._decorate(
            code, cls.DELIMITER.join(map(cls._to_string, value))
        )

    @classmethod
    def _decorate(cls, code: Code, string: str) -> str:
        """
        Add SOH, header, SOT and EOT to the string.

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
        header = cls.SOH + cls._HEADERS_R[code]
        return header + cls.SOT + string + cls.EOT

    @classmethod
    def _find_border(cls, string: str) -> tuple[int, int]:
        """
        find SOT and EOT for SOH at the beginning of the string.

        Parameters
        ----------
        string: str
            string.

        Returns
        -------
        tuple[int, int]
            SOT and EOT positions.
        """
        assert cls._soh_exists(string), "SOH not exists: %s" % string
        sot_pos, opened_sot = -1, 0
        for i_char in range(len(string)):
            char = string[i_char]

            if char == cls.SOT:
                opened_sot += 1
                if sot_pos < 0:
                    sot_pos = i_char

            elif char == cls.EOT:
                opened_sot -= 1

            if not opened_sot and sot_pos > 0:
                return sot_pos, i_char

        raise ValueError("SOV not closed in %r" % string)

    @classmethod
    def _iter_string(cls, string: str) -> Generator[str, None, None]:
        """
        Iterate string by values.

        Parameters
        ----------
        string: str
            string for iterating.

        Yields
        ------
        str
            single raw value.
        """
        length, i_char, raw = len(string), 0, ""
        while i_char < length:
            char = string[i_char]

            if char == cls.SOH:
                assert not len(raw), "raw value not empty"
                _, eot_pos = cls._find_border(string[i_char:])
                yield string[i_char:i_char + eot_pos + 1]
                i_char += eot_pos + 1

            elif char == cls.DELIMITER:
                yield raw
                raw = ""

            else:
                raw += char
            i_char += 1

        if len(raw) and len(string):
            yield raw

    @classmethod
    def _read_header(cls, string: str) -> tuple[Code, str]:
        """
        Read header and get Code with clear content.

        Parameters
        ----------
        string: str
            raw string

        Returns
        -------
        tuple[Code, str]
            header code and clear content.
        """
        code = cls.HEADERS[string[1:4]]
        sot_pos, eot_pos = cls._find_border(string)
        clear_value = string[sot_pos + 1:eot_pos]
        return code, clear_value

    @classmethod
    def _soh_exists(cls, string: str) -> bool:
        """Check that SOH exists in the string."""
        return (
            len(string) >= 6
            and string[0] == cls.SOH
            and cls.SOT in string
            and cls.EOT in string
            and string[1:4] in cls.HEADERS
        )

    @classmethod
    def _to_string(cls, val: Any) -> str:
        """Convert value to string."""
        if isinstance(val, cls.ITERS | str):
            return cls.encode(val)
        elif isinstance(val, Code):
            val = val.value
        return str(val)

    @classmethod
    def _to_value(cls, val: str) -> Any:
        """
        Convert the value from a string (if possible)
        or return the value as is.
        """
        if not isinstance(val, str) or not len(val):
            return val

        elif cls._soh_exists(val):
            return cls.decode(val)

        elif cls.INT.match(val) is not None:
            return int(val)

        elif cls.FLOAT.match(val) is not None:
            return float(val)

        elif val in ("True", "False", "None"):
            return None if val == "None" else val == "True"

        else:
            return val


class BytesEncoder(object):

    TYPES = {
        Code.BYTES,
    }

    @classmethod
    def decode(cls, value: bytes, type_: Code) -> Any:
        """
        Decode value from bytes.

        Parameters
        ----------
        value: bytes
            value encoded in the bytes.
        type_: Code
            indicate what type expected in bytes.

        Returns
        -------
        Any
            decoded value.
        """
        ...

    @classmethod
    def encode(cls, value: Any, type_: Code) -> bytes:
        """
        Encode value to the bytes.

        Parameters
        ----------
        value: Any
            value for encoding.
        type_: Code
            indicate what type expected in bytes.

        Returns
        -------
        bytes
            encoded value.
        """
        ...

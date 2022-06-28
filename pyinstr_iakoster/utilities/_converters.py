import re
import itertools
from typing import Any

import numpy as np

from ..core import Code


__all__ = [
    "StringConverter",
]


class StringConverter(object):

    SINGLE_TYPES = int | float | str | bool | None
    ITERS = dict | list | np.ndarray | set | tuple

    DELIMITER = ","
    PARAMETER = "="

    INT = re.compile("^\d+$")
    FLOAT = re.compile("^\d+\.\d+$")
    EFLOAT = re.compile("^\d\.\d+[eE][+-]\d+$")

    SOH = "/"           # start of heading
    STX = "\t"          # start of text
    SOV = f"{SOH}v("    # start of value
    EOV = ")"           # end of value

    HEADERS = {
        "dct": Code.DICT,
        "lst": Code.LIST,
        "npa": Code.NUMPY_ARRAY,
        "set": Code.SET,
        "str": Code.STRING,
        "tpl": Code.TUPLE,
    }
    _HEADERS_R = {v: k for k, v in HEADERS.items()}
    _CONVERTERS = {
        Code.DICT: lambda v: {i: next(v) for i in v},
        Code.LIST: list,
        Code.NUMPY_ARRAY: lambda v: np.array(tuple(v)),
        Code.SET: set,
        Code.TUPLE: tuple,
    }
    _TYPES = {
        dict: Code.DICT,
        list: Code.LIST,
        np.ndarray: Code.NUMPY_ARRAY,
        set: Code.SET,
        tuple: Code.TUPLE
    }

    @classmethod
    def from_str(cls, value: str) -> Any:
        """
        Convert string value to any type.

        If there are no templates to convert, then returns
        the value 'as is' as a string.

        Parameters
        ----------
        value: str
            string value.

        Returns
        -------
        Any
            Converted value.
        """

        def eoh_exists(string: str) -> bool:
            if len(string) < 5 or string[0] != cls.SOH:
                return False
            if cls.STX not in string:
                return False
            return string[1:4] in cls.HEADERS

        def read_header(string: str):
            code_ = cls.HEADERS[string[1:4]]
            eoh_pos = string.find(cls.STX)
            assert eoh_pos != -1, "STX not found"
            return code_, {}, string[eoh_pos + 1:]

        def chain_map(string: str):
            length, i_ch, raw = len(string), 0, ""
            while i_ch < length:
                ch = string[i_ch]
                if ch == cls.SOH:
                    assert not len(raw), "raw value not empty"
                    end = find_value_end(string[i_ch:])
                    yield cls.from_str(string[i_ch + 3:i_ch + end])
                    i_ch += end + 1
                elif ch == cls.DELIMITER:
                    yield raw
                    raw = ""
                else:
                    raw += ch
                i_ch += 1

            if len(raw) and len(string):
                yield raw

        def find_value_end(val: str):
            assert val[:3] == cls.SOV, "SOV not found"
            opened_sov = 1
            for i_ch in range(3, len(val)):
                if val[i_ch:i_ch + 3] == cls.SOV:
                    opened_sov += 1
                elif val[i_ch] == cls.EOV:
                    opened_sov -= 1
                if not opened_sov:
                    return i_ch
            raise ValueError("SOV not closed")

        def to_value(val: Any | str) -> Any:
            if not isinstance(val, str):
                return val

            elif val[:3] == cls.SOV and len(val) < 9:
                return cls.from_str(val[find_value_end(val)])

            elif cls.INT.match(val) is not None:
                return int(val)

            elif (cls.FLOAT.match(val) or cls.EFLOAT.match(val)) is not None:
                return float(val)

            elif val in ("True", "False", "None"):
                return None if val == "None" else val == "True"

            else:
                return val

        if eoh_exists(value):
            code, pars, value = read_header(value)
            if code is Code.STRING:
                return value
            #print(list(map(to_value, chain_map(value))))
            return cls._CONVERTERS[code](map(to_value, chain_map(value)))
        return to_value(value)

    @classmethod
    def to_str(cls, value: Any) -> str:
        """
        Convert any value to the string.

        There is a specific rules for converting
        dict, tuple and list.

        Parameters
        ----------
        value: Any
            value for converting.

        Returns
        -------
        str
            value as str
        """

        def add_header(code_: Code, string: str, **pars: Any) -> str:
            code_key = cls._HEADERS_R[code_]
            assert not len(pars), "parameters is not supported yet"
            pars = ""
            return "{}{}{}{}{}".format(cls.SOH, code_key, pars, cls.STX, string)

        def to_string(val: Any) -> str:
            if isinstance(val, cls.ITERS):
                return cls.SOV + cls.to_str(val) + cls.EOV
            return str(val)

        def prepare_value(val):
            if isinstance(val, dict):
                val = itertools.chain.from_iterable(value.items())
            return cls._TYPES[type(value)], val

        if isinstance(value, str):
            return add_header(Code.STRING, value)
        if type(value) not in cls._TYPES:
            return to_string(value)

        code, value = prepare_value(value)
        return add_header(code, cls.DELIMITER.join(map(to_string, value)))

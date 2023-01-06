from enum import IntEnum, auto


__all__ = [
    "Code"
]


class Code(IntEnum):  # nodesc

    # System codes
    NONE = 0
    OK = auto()
    WAIT = auto()
    ERROR = auto()
    RAISE = auto()
    UNDEFINED = 0xff

    # Type codes
    BYTES = 0x100
    DICT = auto()
    LIST = auto()
    NUMPY_ARRAY = auto()
    SET = auto()
    STRING = auto()
    TUPLE = auto()

from enum import IntEnum, auto


__all__ = [
    "Code"
]


class Code(IntEnum):

    # System codes
    NONE = -1

    # Type codes
    BYTES = 0x100
    DICT = auto()
    LIST = auto()
    NUMPY_ARRAY = auto()
    SET = auto()
    STRING = auto()
    TUPLE = auto()

    # status codes
    OK = 0x500
    WAIT = auto()
    ERROR = auto()
    RAISE = auto()
    UNDEFINED = 0x5ff

from enum import IntEnum, auto


__all__ = [
    "Code"
]


class Code(IntEnum):  # nodesc

    # System codes
    NONE = 0
    UNDEFINED = 0xff

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

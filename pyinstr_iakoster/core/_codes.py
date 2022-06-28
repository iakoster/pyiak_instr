from enum import IntEnum, auto


__all__ = [
    "Code"
]


class Code(IntEnum):

    # System codes
    NONE = -1

    # Type codes
    BYTES = 0x0100
    DICT = auto()
    LIST = auto()
    NUMPY_ARRAY = auto()
    SET = auto()
    STRING = auto()
    TUPLE = auto()

"""Private module of ``pyiak_instr.core`` with codes."""
from enum import IntEnum, auto


__all__ = ["Code"]


class Code(IntEnum):
    """
    Represents the codes that are used in the library.
    """

    # System codes
    NONE = 0
    OK = auto()
    WAIT = auto()
    ERROR = auto()
    UNDEFINED = 0xFF

    # Type codes
    BOOL = 0x100
    BYTES = auto()
    DICT = auto()
    FLOAT = auto()
    INT = auto()
    LIST = auto()
    SET = auto()
    STRING = auto()
    TUPLE = auto()
    NUMPY_ARRAY = auto()

    # Additional types
    WORDS = 0x200

    # Additional errors
    INVALID_ID = 0x300

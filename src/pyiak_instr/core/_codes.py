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
    DEFAULT = auto()
    ANY = auto()
    UNDEFINED = 0xFF

    # Type codes
    BOOL = 0x100
    BYTES = auto()
    CODE = auto()
    DICT = auto()
    FLOAT = auto()
    INT = auto()
    LIST = auto()
    SET = auto()
    STRING = auto()
    TUPLE = auto()
    NUMPY_ARRAY = auto()

    # Value types codes
    I8 = 0x200
    I16 = auto()
    I24 = auto()
    I32 = auto()
    I40 = auto()
    I48 = auto()
    I56 = auto()
    I64 = auto()
    U8 = auto()
    U16 = auto()
    U24 = auto()
    U32 = auto()
    U40 = auto()
    U48 = auto()
    U56 = auto()
    U64 = auto()
    F16 = auto()
    F32 = auto()
    F64 = auto()
    CHAR = auto()

    # Additional types
    WORDS = 0x300

    # Additional errors
    INVALID_ID = 0x400

    # Additional codes
    BIG_ENDIAN = 0x500
    LITTLE_ENDIAN = auto()

    # Common codes
    ACTUAL = 0x600
    READ = auto()
    WRITE = auto()
    DMA = auto()
    STRONG = auto()
    EXPECTED = auto()
    BASIC = auto()
    SINGLE = auto()
    STATIC = auto()
    ADDRESS = auto()
    CRC = auto()
    DATA = auto()
    DYNAMIC_LENGTH = auto()
    ID = auto()
    OPERATION = auto()
    RESPONSE = auto()
    READ_ONLY = auto()
    WRITE_ONLY = auto()
    RX = auto()
    TX = auto()
    SELF = auto()

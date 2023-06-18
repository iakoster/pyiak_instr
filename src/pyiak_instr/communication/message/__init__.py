"""
==========================================
Message (:mod:`pyiak_instr.communication`)
==========================================
"""
# pylint: disable=duplicate-code
from ._struct import (
    STRUCT_DATACLASS,
    Basic,
    Static,
    Address,
    Crc,
    Data,
    DynamicLength,
    Id,
    Operation,
    Response,
    FieldUnionT,
    StructGetParser,
    StructHasParser,
    Struct,
)
from ._message import Message
from ._pattern import (
    FieldPattern,
    StructPattern,
    MessagePattern,
)

__all__ = [
    "STRUCT_DATACLASS",
    "Message",
    "Basic",
    "FieldUnionT",
    "Static",
    "Address",
    "Crc",
    "Data",
    "DynamicLength",
    "Id",
    "Operation",
    "Response",
    "StructGetParser",
    "StructHasParser",
    "Struct",
    "FieldPattern",
    "StructPattern",
    "MessagePattern",
]

"""
================================================
Types (:mod:`pyiak_instr.communication.message`)
================================================
"""
# pylint: disable=duplicate-code
from ._pattern import (
    FieldPattern,
    StructPattern,
    MessagePattern,
)
from ._message import Message
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

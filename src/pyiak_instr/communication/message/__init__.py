"""
==========================================
Message (:mod:`pyiak_instr.communication`)
==========================================
"""
# pylint: disable=duplicate-code
from ._struct import (
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

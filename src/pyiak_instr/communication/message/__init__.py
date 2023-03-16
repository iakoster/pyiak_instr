"""
==========================================
Message (:mod:`pyiak_instr.communication`)
==========================================
"""
# pylint: disable=duplicate-code
from ._field import (
    MessageField,
    SingleMessageField,
    StaticMessageField,
    AddressMessageField,
    CrcMessageField,
    DataMessageField,
    DataLengthMessageField,
    IdMessageField,
    OperationMessageField,
    ResponseMessageField,
    MessageFieldUnionT,
    MessageFieldPattern,
)
from ._message import MessageFieldParser


__all__ = [
    "MessageField",
    "SingleMessageField",
    "StaticMessageField",
    "AddressMessageField",
    "CrcMessageField",
    "DataMessageField",
    "DataLengthMessageField",
    "IdMessageField",
    "OperationMessageField",
    "ResponseMessageField",
    "MessageFieldUnionT",
    "MessageFieldPattern",
    "MessageFieldParser",
]

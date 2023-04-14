"""
==========================================
Message (:mod:`pyiak_instr.communication`)
==========================================
"""
# pylint: disable=duplicate-code
from ._struct import (
    MessageFieldStruct,
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DataLengthMessageFieldStruct,
    IdMessageFieldStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
    MessageFieldStructUnionT,
)


__all__ = [
    "MessageFieldStruct",
    "SingleMessageFieldStruct",
    "StaticMessageFieldStruct",
    "AddressMessageFieldStruct",
    "CrcMessageFieldStruct",
    "DataMessageFieldStruct",
    "DataLengthMessageFieldStruct",
    "IdMessageFieldStruct",
    "OperationMessageFieldStruct",
    "ResponseMessageFieldStruct",
    "MessageFieldStructUnionT",
]

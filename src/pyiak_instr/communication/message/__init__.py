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
    MessageFieldPattern,
)
from ._field import (
    MessageFieldABC,
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
    "MessageFieldPattern",
    "MessageFieldABC",
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
]

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
from ._message import Message, MessagePattern


__all__ = [
    "AddressMessageField",
    "AddressMessageFieldStruct",
    "CrcMessageField",
    "CrcMessageFieldStruct",
    "DataLengthMessageField",
    "DataLengthMessageFieldStruct",
    "DataMessageField",
    "DataMessageFieldStruct",
    "IdMessageField",
    "IdMessageFieldStruct",
    "Message",
    "MessageField",
    "MessageFieldPattern",
    "MessageFieldStruct",
    "MessageFieldStructUnionT",
    "MessageFieldUnionT",
    "MessagePattern",
    "OperationMessageField",
    "OperationMessageFieldStruct",
    "ResponseMessageField",
    "ResponseMessageFieldStruct",
    "SingleMessageField",
    "SingleMessageFieldStruct",
    "StaticMessageField",
    "StaticMessageFieldStruct",
]

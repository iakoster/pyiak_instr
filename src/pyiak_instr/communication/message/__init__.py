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
    MessageStruct,
)
from ._message import Message
from ._pattern import (
    MessageFieldStructPattern,
    MessageStructPattern,
    MessagePattern,
)


__all__ = [
    "AddressMessageFieldStruct",
    "CrcMessageFieldStruct",
    "DataLengthMessageFieldStruct",
    "DataMessageFieldStruct",
    "IdMessageFieldStruct",
    "Message",
    "MessageFieldStruct",
    "MessageFieldStructPattern",
    "MessageFieldStructUnionT",
    "MessagePattern",
    "MessageStruct",
    "MessageStructPattern",
    "OperationMessageFieldStruct",
    "ResponseMessageFieldStruct",
    "SingleMessageFieldStruct",
    "StaticMessageFieldStruct",
]

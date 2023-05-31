"""Private module of ``pyiak_instr.communication.message``"""
from __future__ import annotations
from dataclasses import field as _field
from typing import Union

from ...core import Code
from .types import (
    STRUCT_DATACLASS,
    MessageFieldStructABC,
    StaticMessageFieldStructABC,
    AddressMessageFieldStructABC,
    CrcMessageFieldStructABC,
    DataMessageFieldStructABC,
    DynamicLengthMessageFieldStructABC,
    IdMessageFieldStructABC,
    OperationMessageFieldStructABC,
    ResponseMessageFieldStructABC,
    MessageStructABC,
)


__all__ = [
    "MessageFieldStruct",
    "StaticMessageFieldStruct",
    "AddressMessageFieldStruct",
    "CrcMessageFieldStruct",
    "DataMessageFieldStruct",
    "DynamicLengthMessageFieldStruct",
    "IdMessageFieldStruct",
    "OperationMessageFieldStruct",
    "ResponseMessageFieldStruct",
    "MessageFieldStructUnionT",
    "MessageStruct",
]


@STRUCT_DATACLASS
class MessageFieldStruct(MessageFieldStructABC):
    """Represents a general field of a Message."""


@STRUCT_DATACLASS
class StaticMessageFieldStruct(StaticMessageFieldStructABC):
    """
    Represents a field of a Message with static single word (e.g. preamble).
    """


@STRUCT_DATACLASS
class AddressMessageFieldStruct(AddressMessageFieldStructABC):
    """
    Represents a field of a Message with address.
    """


@STRUCT_DATACLASS
class CrcMessageFieldStruct(CrcMessageFieldStructABC):
    """
    Represents a field of a Message with crc value.
    """


@STRUCT_DATACLASS
class DataMessageFieldStruct(DataMessageFieldStructABC):
    """Represents a field of a Message with data."""


@STRUCT_DATACLASS
class DynamicLengthMessageFieldStruct(DynamicLengthMessageFieldStructABC):
    """
    Represents a field of a Message with data length.
    """


@STRUCT_DATACLASS
class IdMessageFieldStruct(IdMessageFieldStructABC):
    """
    Represents a field with a unique identifier of a particular message.
    """


@STRUCT_DATACLASS
class OperationMessageFieldStruct(OperationMessageFieldStructABC):
    """
    Represents a field of a Message with operation (e.g. read).
    """


@STRUCT_DATACLASS
class ResponseMessageFieldStruct(ResponseMessageFieldStructABC):
    """
    Represents a field of a Message with response field.
    """


MessageFieldStructUnionT = Union[  # pylint: disable=invalid-name
    MessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DynamicLengthMessageFieldStruct,
    IdMessageFieldStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
]


@STRUCT_DATACLASS
class MessageStruct(MessageStructABC[MessageFieldStructUnionT]):
    """
    Message for communication between devices.
    """

    _field_type_codes: dict[type[MessageFieldStructUnionT], Code] = _field(
        default_factory=lambda: {
            MessageFieldStruct: Code.BASIC,
            StaticMessageFieldStruct: Code.STATIC,
            AddressMessageFieldStruct: Code.ADDRESS,
            CrcMessageFieldStruct: Code.CRC,
            DataMessageFieldStruct: Code.DATA,
            DynamicLengthMessageFieldStruct: Code.DYNAMIC_LENGTH,
            IdMessageFieldStruct: Code.ID,
            OperationMessageFieldStruct: Code.OPERATION,
            ResponseMessageFieldStruct: Code.RESPONSE,
        },
        init=False,
    )

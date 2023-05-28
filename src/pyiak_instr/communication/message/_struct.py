"""Private module of ``pyiak_instr.communication.message``"""
from __future__ import annotations
from dataclasses import field as _field
from typing import Union

from ...core import Code
from .types import (
    STRUCT_DATACLASS,
    MessageFieldStructABC,
    SingleMessageFieldStructABC,
    StaticMessageFieldStructABC,
    AddressMessageFieldStructABC,
    CrcMessageFieldStructABC,
    DataMessageFieldStructABC,
    DataLengthMessageFieldStructABC,
    IdMessageFieldStructABC,
    OperationMessageFieldStructABC,
    ResponseMessageFieldStructABC,
    MessageStructABC,
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
    "MessageStruct",
]


@STRUCT_DATACLASS
class MessageFieldStruct(MessageFieldStructABC):
    """Represents a general field of a Message."""


@STRUCT_DATACLASS
class SingleMessageFieldStruct(SingleMessageFieldStructABC):
    """
    Represents a field of a Message with single word.
    """


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
class DataLengthMessageFieldStruct(DataLengthMessageFieldStructABC):
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
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DataLengthMessageFieldStruct,
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
            SingleMessageFieldStruct: Code.SINGLE,
            StaticMessageFieldStruct: Code.STATIC,
            AddressMessageFieldStruct: Code.ADDRESS,
            CrcMessageFieldStruct: Code.CRC,
            DataMessageFieldStruct: Code.DATA,
            DataLengthMessageFieldStruct: Code.DATA_LENGTH,
            IdMessageFieldStruct: Code.ID,
            OperationMessageFieldStruct: Code.OPERATION,
            ResponseMessageFieldStruct: Code.RESPONSE,
        },
        init=False,
    )

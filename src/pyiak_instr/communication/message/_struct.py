from __future__ import annotations
from dataclasses import InitVar, field as _field
from typing import Union

from ...core import Code
from ...encoders import BytesEncoder
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
]


@STRUCT_DATACLASS
class MessageFieldStruct(MessageFieldStructABC):
    """Represents a general field of a Message."""

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class SingleMessageFieldStruct(SingleMessageFieldStructABC):
    """
    Represents a field of a Message with single word.
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class StaticMessageFieldStruct(StaticMessageFieldStructABC):
    """
    Represents a field of a Message with static single word (e.g. preamble).
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class AddressMessageFieldStruct(AddressMessageFieldStructABC):
    """
    Represents a field of a Message with address.
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class CrcMessageFieldStruct(CrcMessageFieldStructABC):
    """
    Represents a field of a Message with crc value.
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class DataMessageFieldStruct(DataMessageFieldStructABC):
    """Represents a field of a Message with data."""

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class DataLengthMessageFieldStruct(DataLengthMessageFieldStructABC):
    """
    Represents a field of a Message with data length.
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class IdMessageFieldStruct(IdMessageFieldStructABC):
    """
    Represents a field with a unique identifier of a particular message.
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class OperationMessageFieldStruct(OperationMessageFieldStructABC):
    """
    Represents a field of a Message with operation (e.g. read).
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class ResponseMessageFieldStruct(ResponseMessageFieldStructABC):
    """
    Represents a field of a Message with response field.
    """

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


MessageFieldStructUnionT = Union[
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


#
#
# class MessageFieldPattern(MessageFieldStructPatternABC[MessageFieldStructUnionT]):
#     """
#     Represents pattern for message field struct
#     """
#
#     _options = {
#         "basic": MessageFieldStruct,
#         "single": SingleMessageFieldStruct,
#         "static": StaticMessageFieldStruct,
#         "address": AddressMessageFieldStruct,
#         "crc": CrcMessageFieldStruct,
#         "data": DataMessageFieldStruct,
#         "data_length": DataLengthMessageFieldStruct,
#         "id": IdMessageFieldStruct,
#         "operation": OperationMessageFieldStruct,
#         "response": ResponseMessageFieldStruct,
#     }
#
#     @staticmethod
#     def get_bytesize(fmt: Code) -> int:
#         """
#         Get fmt size in bytes.
#
#         Parameters
#         ----------
#         fmt : Code
#             fmt code.
#
#         Returns
#         -------
#         int
#             fmt bytesize.
#         """
#         return BytesEncoder.get_bytesize(fmt)

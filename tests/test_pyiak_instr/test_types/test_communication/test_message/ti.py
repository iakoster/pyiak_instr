from dataclasses import InitVar

from src.pyiak_instr.core import Code
from src.pyiak_instr.encoders import BytesEncoder
from src.pyiak_instr.types.communication.message import (
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
    MessageABC,
)


@STRUCT_DATACLASS
class TIMessageFieldStruct(MessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TISingleMessageFieldStruct(SingleMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIStaticMessageFieldStruct(StaticMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIAddressMessageFieldStruct(AddressMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TICrcMessageFieldStruct(CrcMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIDataMessageFieldStruct(DataMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIDataLengthMessageFieldStruct(DataLengthMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIIdMessageFieldStruct(IdMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIOperationMessageFieldStruct(OperationMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIResponseMessageFieldStruct(ResponseMessageFieldStructABC):
    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIMessageStruct(MessageStructABC):
    ...


class TIMessage(MessageABC):
    _field_codes = {
        Code.BASIC: TIMessageFieldStruct,
        Code.SINGLE: TISingleMessageFieldStruct,
        Code.STATIC: TIStaticMessageFieldStruct,
        Code.ADDRESS: TIAddressMessageFieldStruct,
        Code.CRC: TICrcMessageFieldStruct,
        Code.DATA: TIDataMessageFieldStruct,
        Code.DATA_LENGTH: TIDataLengthMessageFieldStruct,
        Code.ID: TIIdMessageFieldStruct,
        Code.OPERATION: TIOperationMessageFieldStruct,
        Code.RESPONSE: TIResponseMessageFieldStruct,
    }

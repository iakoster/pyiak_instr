from dataclasses import InitVar, field as _field
from typing import Union

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
    MessageFieldStructPatternABC,
    MessageStructPatternABC,
    MessagePatternABC,
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


TIFieldStructUnionT = Union[
    TIMessageFieldStruct,
    TISingleMessageFieldStruct,
    TIStaticMessageFieldStruct,
    TIAddressMessageFieldStruct,
    TICrcMessageFieldStruct,
    TIDataMessageFieldStruct,
    TIDataLengthMessageFieldStruct,
    TIIdMessageFieldStruct,
    TIOperationMessageFieldStruct,
    TIResponseMessageFieldStruct,
]


@STRUCT_DATACLASS
class TIMessageStruct(MessageStructABC):

    _field_type_codes: dict[type[TIFieldStructUnionT], Code] = _field(
        default_factory=lambda: {
            TIMessageFieldStruct: Code.BASIC,
            TISingleMessageFieldStruct: Code.SINGLE,
            TIStaticMessageFieldStruct: Code.STATIC,
            TIAddressMessageFieldStruct: Code.ADDRESS,
            TICrcMessageFieldStruct: Code.CRC,
            TIDataMessageFieldStruct: Code.DATA,
            TIDataLengthMessageFieldStruct: Code.DATA_LENGTH,
            TIIdMessageFieldStruct: Code.ID,
            TIOperationMessageFieldStruct: Code.OPERATION,
            TIResponseMessageFieldStruct: Code.RESPONSE,
        },
        init=False,
    )


class TIMessage(MessageABC):
    ...


class TIMessageFieldStructPattern(
    MessageFieldStructPatternABC[TIFieldStructUnionT]
):

    _options = dict(
        basic=TIMessageFieldStruct,
        single=TISingleMessageFieldStruct,
        static=TIStaticMessageFieldStruct,
        address=TIAddressMessageFieldStruct,
        crc=TICrcMessageFieldStruct,
        data=TIDataMessageFieldStruct,
        data_length=TIDataLengthMessageFieldStruct,
        id=TIIdMessageFieldStruct,
        operation=TIOperationMessageFieldStruct,
        response=TIResponseMessageFieldStruct,
    )


class TIMessageStructPattern(
    MessageStructPatternABC[TIMessageStruct, TIMessageFieldStructPattern]
):
    _options = {"basic": TIMessageStruct}


class TIMessagePattern(
    MessagePatternABC[TIMessage, TIMessageStructPattern]
):
    _options = {"basic": TIMessage}

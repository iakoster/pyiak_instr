from dataclasses import InitVar, field as _field
from typing import Union

from src.pyiak_instr.core import Code
from src.pyiak_instr.encoders import BytesEncoder
from src.pyiak_instr.communication.message.types import (
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
    MessageABC,
    MessageFieldStructPatternABC,
    MessageStructPatternABC,
    MessagePatternABC,
)


@STRUCT_DATACLASS
class TIMessageFieldStruct(MessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIStaticMessageFieldStruct(StaticMessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIAddressMessageFieldStruct(AddressMessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TICrcMessageFieldStruct(CrcMessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIDataMessageFieldStruct(DataMessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIDynamicLengthMessageFieldStruct(DynamicLengthMessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIIdMessageFieldStruct(IdMessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIOperationMessageFieldStruct(OperationMessageFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIResponseMessageFieldStruct(ResponseMessageFieldStructABC):
    ...


TIFieldStructUnionT = Union[
    TIMessageFieldStruct,
    TIStaticMessageFieldStruct,
    TIAddressMessageFieldStruct,
    TICrcMessageFieldStruct,
    TIDataMessageFieldStruct,
    TIDynamicLengthMessageFieldStruct,
    TIIdMessageFieldStruct,
    TIOperationMessageFieldStruct,
    TIResponseMessageFieldStruct,
]


@STRUCT_DATACLASS
class TIMessageStruct(MessageStructABC):

    _field_type_codes: dict[type[TIFieldStructUnionT], Code] = _field(
        default_factory=lambda: {
            TIMessageFieldStruct: Code.BASIC,
            TIStaticMessageFieldStruct: Code.STATIC,
            TIAddressMessageFieldStruct: Code.ADDRESS,
            TICrcMessageFieldStruct: Code.CRC,
            TIDataMessageFieldStruct: Code.DATA,
            TIDynamicLengthMessageFieldStruct: Code.DYNAMIC_LENGTH,
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
        static=TIStaticMessageFieldStruct,
        address=TIAddressMessageFieldStruct,
        crc=TICrcMessageFieldStruct,
        data=TIDataMessageFieldStruct,
        data_length=TIDynamicLengthMessageFieldStruct,
        id=TIIdMessageFieldStruct,
        operation=TIOperationMessageFieldStruct,
        response=TIResponseMessageFieldStruct,
    )

    @staticmethod
    def get_fmt_bytesize(fmt: Code) -> int:
        return BytesEncoder(fmt=fmt).value_size


class TIMessageStructPattern(
    MessageStructPatternABC[TIMessageStruct, TIMessageFieldStructPattern]
):
    _options = {"basic": TIMessageStruct}


class TIMessagePattern(
    MessagePatternABC[TIMessage, TIMessageStructPattern]
):
    _options = {"basic": TIMessage}

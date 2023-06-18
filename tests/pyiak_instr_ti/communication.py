from __future__ import annotations
from dataclasses import field as _field
from typing import Union

from src.pyiak_instr.core import Code
from src.pyiak_instr.encoders import BytesEncoder
from src.pyiak_instr.communication.message import (
    STRUCT_DATACLASS,
    Basic,
    Static,
    Address,
    Crc,
    Data,
    DynamicLength,
    Id,
    Operation,
    Response,
    Struct,
    Message,
    FieldPattern,
    StructPattern,
    MessagePattern,
)


@STRUCT_DATACLASS
class TIBasic(Basic):
    ...


@STRUCT_DATACLASS
class TIStatic(Static):
    ...


@STRUCT_DATACLASS
class TIAddress(Address):
    ...


@STRUCT_DATACLASS
class TICrc(Crc):
    ...


@STRUCT_DATACLASS
class TIData(Data):
    ...


@STRUCT_DATACLASS
class TIDynamicLength(DynamicLength):
    ...


@STRUCT_DATACLASS
class TIId(Id):
    ...


@STRUCT_DATACLASS
class TIOperation(Operation):
    ...


@STRUCT_DATACLASS
class TIResponse(Response):
    ...


TIFieldUnionT = Union[
    TIBasic,
    TIStatic,
    TIAddress,
    TICrc,
    TIData,
    TIDynamicLength,
    TIId,
    TIOperation,
    TIResponse,
]


@STRUCT_DATACLASS
class TIStruct(Struct):

    _field_type_codes: dict[type[TIFieldUnionT], Code] = _field(
        default_factory=lambda: {
            TIBasic: Code.BASIC,
            TIStatic: Code.STATIC,
            TIAddress: Code.ADDRESS,
            TICrc: Code.CRC,
            TIData: Code.DATA,
            TIDynamicLength: Code.DYNAMIC_LENGTH,
            TIId: Code.ID,
            TIOperation: Code.OPERATION,
            TIResponse: Code.RESPONSE,
        },
        init=False,
    )


class TIMessage(Message):
    ...


class TIFieldPattern(
    FieldPattern[TIFieldUnionT]
):

    _options = dict(
        basic=TIBasic,
        static=TIStatic,
        address=TIAddress,
        crc=TICrc,
        data=TIData,
        dynamic_length=TIDynamicLength,
        id=TIId,
        operation=TIOperation,
        response=TIResponse,
    )

    @staticmethod
    def get_fmt_bytesize(fmt: Code) -> int:
        return BytesEncoder(fmt=fmt).value_size


class TIStructPattern(
    StructPattern[TIStruct, TIFieldPattern]
):
    _sub_p_type = TIFieldPattern
    _options = {"basic": TIStruct}


class TIMessagePattern(
    MessagePattern[TIMessage, TIStructPattern]
):
    _sub_p_type = TIStructPattern
    _options = {"basic": TIMessage}

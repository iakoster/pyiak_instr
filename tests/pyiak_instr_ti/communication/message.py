from __future__ import annotations
from dataclasses import field as _field
from typing import Union

from src.pyiak_instr.core import Code
from src.pyiak_instr.communication.message import (
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


class TIBasic(Basic):
    ...


class TIStatic(Static):
    ...


class TIAddress(Address):
    ...


class TICrc(Crc):
    ...


class TIData(Data):
    ...


class TIDynamicLength(DynamicLength):
    ...


class TIId(Id):
    ...


class TIOperation(Operation):
    ...


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


class TIStruct(Struct):

    _f_type_codes: dict[type[TIFieldUnionT], Code] = {
            TIBasic: Code.BASIC,
            TIStatic: Code.STATIC,
            TIAddress: Code.ADDRESS,
            TICrc: Code.CRC,
            TIData: Code.DATA,
            TIDynamicLength: Code.DYNAMIC_LENGTH,
            TIId: Code.ID,
            TIOperation: Code.OPERATION,
            TIResponse: Code.RESPONSE,
        }


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

from ...core import Code
from ...encoders import BytesEncoder
from .types import (
    MessageFieldStructPatternABC,
    MessagePatternABC,
    MessageStructPatternABC,
)
from . import (
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataLengthMessageFieldStruct,
    DataMessageFieldStruct,
    IdMessageFieldStruct,
    Message,
    MessageFieldStruct,
    MessageFieldStructUnionT,
    MessageStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
)


__all__ = [
    "MessageFieldStructPattern",
    "MessageStructPattern",
    "MessagePattern",
]


class MessageFieldStructPattern(
    MessageFieldStructPatternABC[MessageFieldStructUnionT]
):
    _options = dict(
        basic=MessageFieldStruct,
        single=SingleMessageFieldStruct,
        static=StaticMessageFieldStruct,
        address=AddressMessageFieldStruct,
        crc=CrcMessageFieldStruct,
        data=DataMessageFieldStruct,
        data_length=DataLengthMessageFieldStruct,
        id=IdMessageFieldStruct,
        operation=OperationMessageFieldStruct,
        response=ResponseMessageFieldStruct,
    )

    @staticmethod
    def get_fmt_bytesize(fmt: Code) -> int:
        return BytesEncoder(fmt=fmt).value_size


class MessageStructPattern(
    MessageStructPatternABC[MessageStruct, MessageFieldStructPattern]
):
    _options = dict(basic=MessageStruct)


class MessagePattern(MessagePatternABC[Message, MessageStructPattern]):
    _options = dict(basic=Message)

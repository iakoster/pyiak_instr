"""Private module of ``pyiak_instr.communication.message``"""
from typing import Any

from ...core import Code
from ...encoders import BytesEncoder
from .types import (
    MessageFieldStructPatternABC,
    MessagePatternABC,
    MessageStructPatternABC,
)
from ._struct import (
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataLengthMessageFieldStruct,
    DataMessageFieldStruct,
    IdMessageFieldStruct,
    MessageFieldStruct,
    MessageFieldStructUnionT,
    MessageStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
)
from ._message import Message


__all__ = [
    "MessageFieldStructPattern",
    "MessageStructPattern",
    "MessagePattern",
]


class MessageFieldStructPattern(
    MessageFieldStructPatternABC[MessageFieldStructUnionT]
):
    """
    Pattern of field struct.
    """

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
        """
        Get fmt size in bytes.

        Parameters
        ----------
        fmt : Code
            fmt code.

        Returns
        -------
        int
            fmt bytesize.
        """
        return BytesEncoder(fmt=fmt).value_size


class MessageStructPattern(
    MessageStructPatternABC[MessageStruct, MessageFieldStructPattern]
):
    """
    Pattern of message struct.
    """

    _options = dict(basic=MessageStruct)


class MessagePattern(MessagePatternABC[Message[Any], MessageStructPattern]):
    """
    Pattern of message.
    """

    _options = dict(basic=Message)

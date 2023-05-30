"""Private module of ``pyiak_instr.communication.message``"""
from typing import Any

from ...core import Code
from ...encoders import BytesEncoder
from ...rwfile import RWConfig
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

    _sub_p_type = MessageFieldStructPattern
    _options = dict(basic=MessageStruct)


class MessagePattern(MessagePatternABC[Message[Any], MessageStructPattern]):
    """
    Pattern of message.
    """

    _rwdata = RWConfig
    _sub_p_type = MessageStructPattern
    _options = dict(basic=Message)

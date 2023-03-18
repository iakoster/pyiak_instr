"""
==========================================
Message (:mod:`pyiak_instr.communication`)
==========================================
"""
# pylint: disable=duplicate-code
from ._field import (
    MessageFieldParameters,
    SingleMessageFieldParameters,
    StaticMessageFieldParameters,
    AddressMessageFieldParameters,
    CrcMessageFieldParameters,
    DataMessageFieldParameters,
    DataLengthMessageFieldParameters,
    IdMessageFieldParameters,
    OperationMessageFieldParameters,
    ResponseMessageFieldParameters,
    MessageFieldUnionT,
    MessageFieldPattern,
)
from ._message import MessageField


__all__ = [
    "MessageFieldParameters",
    "SingleMessageFieldParameters",
    "StaticMessageFieldParameters",
    "AddressMessageFieldParameters",
    "CrcMessageFieldParameters",
    "DataMessageFieldParameters",
    "DataLengthMessageFieldParameters",
    "IdMessageFieldParameters",
    "OperationMessageFieldParameters",
    "ResponseMessageFieldParameters",
    "MessageFieldUnionT",
    "MessageFieldPattern",
    "MessageField",
]

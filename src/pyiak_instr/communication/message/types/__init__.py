"""
================================================
Types (:mod:`pyiak_instr.communication.message`)
================================================
"""
# pylint: disable=duplicate-code
from ._pattern import (
    MessageFieldStructPatternABC,
    MessageStructPatternABC,
    MessagePatternABC,
)
from ._message import MessageABC
from ._struct import (
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
    MessageFieldStructABCUnionT,
    MessageStructGetParser,
    MessageStructHasParser,
    MessageStructABC,
)


__all__ = [
    "STRUCT_DATACLASS",
    "MessageABC",
    "MessageFieldStructABC",
    "MessageFieldStructABCUnionT",
    "SingleMessageFieldStructABC",
    "StaticMessageFieldStructABC",
    "AddressMessageFieldStructABC",
    "CrcMessageFieldStructABC",
    "DataMessageFieldStructABC",
    "DataLengthMessageFieldStructABC",
    "IdMessageFieldStructABC",
    "OperationMessageFieldStructABC",
    "ResponseMessageFieldStructABC",
    "MessageStructGetParser",
    "MessageStructHasParser",
    "MessageStructABC",
    "MessageFieldStructPatternABC",
    "MessageStructPatternABC",
    "MessagePatternABC",
]

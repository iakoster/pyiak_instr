"""
========================================
Communication (:mod:`pyiak_instr.types`)
========================================
"""
# pylint: disable=duplicate-code
from ._message import (
    MessageABC,
    MessageFieldABC,
    MessageGetParserABC,
    MessageHasParserABC,
)


__all__ = [
    "MessageABC",
    "MessageFieldABC",
    "MessageGetParserABC",
    "MessageHasParserABC",
]

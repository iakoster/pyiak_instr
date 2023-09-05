from dataclasses import dataclass

from src.pyiak_instr.communication.format import (
    FormatMap,
    PatternMap,
    Register,
    RegisterMap,
)

from .message import TIMessage, TIMessagePattern


@dataclass(frozen=True, kw_only=True)
class TIRegister(Register):
    ...


class TIRegisterMap(RegisterMap[TIRegister]):
    _register_type = TIRegister


class TIPatternMap(PatternMap[TIMessagePattern]):
    _pattern_type = TIMessagePattern


class TIFormatMap(FormatMap[TIPatternMap, TIRegisterMap, TIMessage]):
    ...

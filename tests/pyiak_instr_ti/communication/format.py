from src.pyiak_instr.store.bin import STRUCT_DATACLASS
from src.pyiak_instr.communication.format import (
    FormatMap,
    PatternMap,
    Register,
    RegisterMap,
)

from .message import TIMessage, TIMessagePattern


@STRUCT_DATACLASS
class TIRegister(Register):
    ...


class TIRegisterMap(RegisterMap[TIRegister]):
    _register_type = TIRegister


class TIPatternMap(PatternMap[TIMessagePattern]):
    _pattern_type = TIMessagePattern


class TIFormatMap(FormatMap[TIPatternMap, TIRegisterMap, TIMessage]):
    ...

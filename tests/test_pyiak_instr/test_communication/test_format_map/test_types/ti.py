from pathlib import Path
from configparser import ConfigParser

from src.pyiak_instr.store.bin.types import STRUCT_DATACLASS
from src.pyiak_instr.communication.message import MessagePattern
from src.pyiak_instr.communication.format_map.types import (
    PatternsMapABC,
    RegisterABC,
    RegistersMapABC,
)


class TIPatternsMap(PatternsMapABC[MessagePattern]):

    _pattern_type = MessagePattern

    @staticmethod
    def _get_pattern_names(path: Path) -> list[str]:
        parser = ConfigParser()
        parser.read(path)
        return parser.sections()


@STRUCT_DATACLASS
class TIRegister(RegisterABC):
    ...


class TIRegistersMap(RegistersMapABC[TIRegister]):

    _register_type = TIRegister

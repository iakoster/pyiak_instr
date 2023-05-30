from pathlib import Path
from configparser import ConfigParser

from src.pyiak_instr.communication.message import MessagePattern
from src.pyiak_instr.communication.format_map.types import (
    PatternsMapABC
)


class TIPatternsMapABC(PatternsMapABC[MessagePattern]):

    _pattern_type = MessagePattern

    @staticmethod
    def _get_pattern_names(path: Path) -> list[str]:
        parser = ConfigParser()
        parser.read(path)
        return parser.sections()
from pathlib import Path
from configparser import ConfigParser

from ..message import MessagePattern
from .types import PatternsMapABC


__all__ = ["PatternsMap"]


class PatternsMap(PatternsMapABC):

    _pattern_type = MessagePattern

    @staticmethod
    def _get_pattern_names(path: Path) -> list[str]:
        parser = ConfigParser()
        parser.read(path)
        return parser.sections()

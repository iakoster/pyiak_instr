"""Private module of ``pyiak_instr.communication.format_map``."""
from pathlib import Path
from configparser import ConfigParser

from ..message import MessagePattern
from .types import PatternsMapABC


__all__ = ["PatternsMap"]


class PatternsMap(PatternsMapABC[MessagePattern]):
    """
    Class which store message patterns.
    """

    _pattern_type = MessagePattern

    @staticmethod
    def _get_pattern_names(path: Path) -> list[str]:
        """
        Get pattern names for file in `path` via configparser.

        Parameters
        ----------
        path : Path
            path to file.

        Returns
        -------
        list[str]
            list of pattern names.
        """
        parser = ConfigParser()
        parser.read(path)
        return parser.sections()

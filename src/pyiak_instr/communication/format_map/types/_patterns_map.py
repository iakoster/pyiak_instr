"""Private module of ``pyiak_instr.communication.format_map.types``."""
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

from ...message.types import MessagePatternABC


__all__ = ["PatternsMapABC"]


MessagePatternT = TypeVar(
    "MessagePatternT", bound=MessagePatternABC[Any, Any]
)


class PatternsMapABC(ABC, Generic[MessagePatternT]):
    """
    Represents base class for message patterns map.
    """

    _pattern_type: type[MessagePatternT]

    def __init__(self, *patterns: MessagePatternT) -> None:
        self._patterns: dict[str, MessagePatternT] = {}
        for pattern in patterns:
            name = pattern.sub_pattern_names[0]
            if name in self._patterns:
                raise KeyError(
                    f"pattern with name {name!r} is already exists"
                )
            self._patterns[name] = pattern

    @staticmethod
    @abstractmethod
    def _get_pattern_names(path: Path) -> list[str]:
        """
        Get pattern names for file in `path`.

        Parameters
        ----------
        path : Path
            path to file.

        Returns
        -------
        list[str]
            list of pattern names.
        """

    def write(self, path: Path) -> None:
        """
        Write patterns to file in `path`.

        Parameters
        ----------
        path : Path
            path to file.
        """
        for pattern in self._patterns.values():
            pattern.write(path)

    @classmethod
    def read(cls, path: Path) -> Self:
        """
        Read patterns from file in `path` and initialize class instance.

        Parameters
        ----------
        path : Path
            path to file.

        Returns
        -------
        Self
            initialized self instance.
        """
        return cls(
            *(
                cls._pattern_type.read(path, n)
                for n in cls._get_pattern_names(path)
            )
        )

    @property
    def pattern_names(self) -> list[str]:
        """
        Returns
        -------
        list[str]
            pattern names.
        """
        return list(self._patterns)

    def __getitem__(self, name: str) -> MessagePatternT:
        """Get pattern by `name`."""
        return self._patterns[name]

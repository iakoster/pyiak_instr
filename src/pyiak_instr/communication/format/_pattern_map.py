"""Private module of ``pyiak_instr.communication.format``."""
from pathlib import Path
from configparser import ConfigParser
from typing import Any, Generic, Self, TypeVar

from ..message import MessagePattern


__all__ = ["PatternMap"]


T = TypeVar("T")
MessagePatternT = TypeVar("MessagePatternT", bound=MessagePattern[Any, Any])


class PatternMap(Generic[MessagePatternT]):
    """
    Represents class for message patterns map.
    """

    _pattern_type: type[MessagePatternT]

    def __init__(self, *patterns: MessagePatternT) -> None:
        self._patterns: dict[str, MessagePatternT] = {}
        for pattern in patterns:
            (name,) = pattern.sub_pattern_names
            if name in self._patterns:
                raise KeyError(
                    f"pattern with name {name!r} is already exists"
                )
            self._patterns[name] = pattern

    def get_pattern(self, name: str) -> MessagePatternT:
        """
        Get pattern by `name`.

        Parameters
        ----------
        name : str
            pattern name.

        Returns
        -------
        MessagePatternT
            message pattern.
        """
        return self._patterns[name]

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

    @property
    def pattern_names(self) -> list[str]:
        """
        Returns
        -------
        list[str]
            pattern names.
        """
        return list(self._patterns)

    def __new__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if not hasattr(cls, "_pattern_type"):
            raise AttributeError(
                f"'{cls.__name__}' object has no attribute 'a'"
            )
        return object.__new__(cls)

"""Private module of ``pyiak_instr`` with common types."""
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self, Optional, Type, TypeVar, Generic


__all__ = ["ContextManager", "WithApi", "WithBaseStringMethods"]


T = TypeVar("T")


class ContextManager(ABC):
    """Represents context manager abstract class."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass


# pylint: disable=too-few-public-methods
class WithApi(ABC, Generic[T]):
    """Represents generic class with some API."""

    def __init__(self, api: T):
        self._api = api

    @property
    def api(self) -> T:
        """
        Returns
        -------
        T
            high-level api instance.
        """
        return self._api


class WithBaseStringMethods(ABC):
    """Represents abstract class with basic string methods."""

    @abstractmethod
    def _get_under_brackets(self) -> str:
        pass

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            String interpretation of the class instance.
        """
        return "<%s>" % self

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            String interpretation of the class instance.
        """
        return "%s(%r)" % (
            self.__class__.__name__,
            self._get_under_brackets(),
        )

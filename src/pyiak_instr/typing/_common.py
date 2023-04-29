"""Private module of ``pyiak_instr`` with common types."""
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Self, Optional, Type, TypeVar, Generic, Protocol


__all__ = [
    "ContextManager",
    "SupportsContainsGetitem",
    "SupportsInitKwargs",
    "WithApi",
    "WithBaseStringMethods",
]


ApiT = TypeVar("ApiT")


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


class SupportsContainsGetitem(Protocol):
    """
    Represents class with `__contains__` and `__getitem__` magic methods.
    """

    def __contains__(self, key: Any) -> bool:
        """Verify that the `key` belongs to the object"""

    def __getitem__(self, key: Any) -> Any:
        """Get item from object by `key`."""


class SupportsInitKwargs(Protocol):
    """
    Represents class with method which returns kwargs for initialization.
    """

    def __init_kwargs__(self) -> dict[str, Any]:
        """Returns kwargs required for initialization."""


class WithApi(ABC, Generic[ApiT]):
    """Represents generic class with some API."""

    def __init__(self, api: ApiT):
        self._api = api

    @property
    def api(self) -> ApiT:
        """
        Returns
        -------
        ApiT
            high-level api instance.
        """
        return self._api


class WithBaseStringMethods(ABC):
    """Represents abstract class with basic string methods."""

    @abstractmethod
    def __str_under_brackets__(self) -> str:
        """
        Returns
        -------
        str
            string for placing under brackets in `.__str__`.
        """

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            String interpretation of the class instance.
        """
        return f"<{self}>"

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            String interpretation of the class instance.
        """
        return f"{self.__class__.__name__}({self.__str_under_brackets__()})"

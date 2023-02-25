from typing import Any


__all__ = ["PyiError"]


class PyiError(Exception):
    """Base class of exceptions for library."""

    def __init__(self, msg: str = "", *args: Any):
        self.msg = msg
        super().__init__(msg, *args)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.msg}"

    def __str__(self) -> str:
        return self.msg

"""Private module of `pyiak_instr.exceptions` with base exceptions."""
from typing import Any


__all__ = ["PyiError"]


class PyiError(Exception):
    """Base class of exceptions for library."""

    def __init__(self, *args: Any, msg: str = ""):
        self.msg = msg
        super().__init__(msg, *args)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.msg}"

    def __str__(self) -> str:
        return self.msg

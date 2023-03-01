"""Private module of `pyiak_instr.exceptions` with common exceptions."""
from ._base import PyiError


__all__ = ["WithoutParent"]


class WithoutParent(PyiError):
    """Raised when further work requires the parent to be specified."""

    def __init__(self) -> None:
        super().__init__(msg="parent not specified")

"""Private module of `pyiak_instr.exceptions` with common exceptions."""
from typing import Any

from ._base import PyiError
from ..core import Code


__all__ = [
    "CodeNotAllowed",
    "NotConfiguredYet",
    "NotAmongTheOptions",
    "WithoutParent",
]


class NotConfiguredYet(PyiError):
    """Raised when further work requires the object to be configured."""

    def __init__(self, obj: object) -> None:
        super().__init__(msg=f"{obj.__class__.__name__} not configured yet")


class NotAmongTheOptions(PyiError):
    """Raised when some parameter not among the options"""

    def __init__(
        self,
        name: str,
        value: Any = None,
        options: set[Any] | None = None,
    ):
        msg = f"{name} option "
        if options is None:
            msg += "not allowed"
        else:
            msg += f"not in {options}"

        if value is not None:
            msg += f", got {repr(value)}"

        super().__init__(msg=msg)


class WithoutParent(PyiError):
    """Raised when further work requires the parent to be specified."""

    def __init__(self) -> None:
        super().__init__(msg="parent not specified")


class CodeNotAllowed(NotAmongTheOptions):
    """Raised when received code not in list of allowed codes."""

    def __init__(self, code: Code, options: set[Code] | None = None) -> None:
        super().__init__("code", value=code, options=options)

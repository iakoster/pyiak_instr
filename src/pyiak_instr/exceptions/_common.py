"""Private module of `pyiak_instr.exceptions` with common exceptions."""
from ._base import PyiError
from ..core import Code


__all__ = ["CodeNotAllowed", "NotConfiguredYet", "WithoutParent"]


class CodeNotAllowed(PyiError):
    """Raised when received code not in list of allowed codes."""

    def __init__(self, code: Code) -> None:
        super().__init__(msg=f"code not allowed: {repr(code)}")


class NotConfiguredYet(PyiError):
    """Raised when further work requires the object to be configured."""

    def __init__(self, obj: object) -> None:
        super().__init__(msg="%s not configured yet" % obj.__class__.__name__)


class WithoutParent(PyiError):
    """Raised when further work requires the parent to be specified."""

    def __init__(self) -> None:
        super().__init__(msg="parent not specified")

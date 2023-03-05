"""Private module of `pyiak_instr.exceptions` with common exceptions."""
from ._base import PyiError


__all__ = ["NotConfiguredYet", "WithoutParent"]


class NotConfiguredYet(PyiError):
    """Raised when further work requires the object to be configured."""

    def __init__(self, obj: object) -> None:
        super().__init__(msg="%s not configured yet" % obj.__class__.__name__)


class WithoutParent(PyiError):
    """Raised when further work requires the parent to be specified."""

    def __init__(self) -> None:
        super().__init__(msg="parent not specified")

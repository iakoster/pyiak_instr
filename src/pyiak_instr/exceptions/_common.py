"""Private module of `pyiak_instr.exceptions` with common exceptions."""
from typing import Any, Iterable

from ._base import PyiError
from ..core import Code


__all__ = [
    "CodeNotAllowed",
    "NotConfiguredYet",
    "NotAmongTheOptions",
    "NotSupportedMethod",
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
        options: Iterable[Any] | None = None,
    ):
        msg = f"'{name}' option "
        if value is not None:
            msg += f"{value!r} "

        if options is None:
            msg += "not allowed"
        else:
            msg += f"not in {{{', '.join(map(repr, options))}}}"

        super().__init__(msg=msg)


class NotSupportedMethod(PyiError):
    """
    Raised when a class method is declared, but not implemented.
    """

    def __init__(self, frame: Any = None) -> None:  # todo: frame type
        msg = ""
        if frame is None:
            msg = "not supported method"

        if hasattr(frame, "f_code"):
            frame = frame.f_code.co_qualname  # type: ignore[union-attr]

        if isinstance(frame, str):
            qualname = frame.split(".")
            if len(qualname) != 2:
                raise ValueError(f"invalid method qualname: {frame!r}")
            msg = f"{qualname[0]} does not support .{qualname[1]}"

        if len(msg) == 0:
            raise TypeError(f"invalid frame type: {frame.__class__}")

        super().__init__(msg=msg)


class WithoutParent(PyiError):
    """Raised when further work requires the parent to be specified."""

    def __init__(self) -> None:
        super().__init__(msg="parent not specified")


class CodeNotAllowed(NotAmongTheOptions):  # todo: useless
    """Raised when received code not in list of allowed codes."""

    def __init__(self, code: Code, options: set[Code] | None = None) -> None:
        super().__init__("code", value=code, options=options)

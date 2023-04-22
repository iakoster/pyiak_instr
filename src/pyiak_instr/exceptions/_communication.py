"""
Private module of `pyiak_instr.exceptions` with exceptions for communication
package.
"""
from typing import Any

from ._base import PyiError


__all__ = ["ContentError"]


class ContentError(PyiError):
    """Raised when any error occurs with the content of"""

    def __init__(self, field: Any, clarification: str = "") -> None:
        msg = f"invalid content in {field.__class__.__name__}"
        if len(clarification) != 0:
            msg += f": {clarification}"
        super().__init__(msg=msg)

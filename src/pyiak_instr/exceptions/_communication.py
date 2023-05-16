"""
Private module of `pyiak_instr.exceptions` with exceptions for communication
package.
"""
from ._base import PyiError


__all__ = ["ContentError"]


class ContentError(PyiError):
    """Raised when any error occurs with the content of"""

    def __init__(self, field: object, clarification: str = "") -> None:
        msg = f"invalid content in {field.__class__.__name__}"  # todo: in/for
        if len(clarification) != 0:
            msg += f": {clarification}"
        super().__init__(msg=msg)

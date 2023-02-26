from ._base import PyiError


__all__ = ["WithoutParent"]


class WithoutParent(PyiError):  # nodesc
    def __init__(self) -> None:
        super().__init__(msg="parent not specified")

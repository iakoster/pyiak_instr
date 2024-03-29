from typing import Any


__all__ = ["PyiError"]


class PyiError(Exception):
    """Base class for pyinstr exceptions."""

    def __init__(self, msg="", *args: Any):
        self.msg = msg
        Exception.__init__(self, msg, *args)

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.msg}"

    def __str__(self):
        return self.msg

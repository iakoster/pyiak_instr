

__all__ = ["PyiError"]


class PyiError(Exception):
    """Base class for pyinstr exceptions."""

    def __init__(self, msg=""):
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__

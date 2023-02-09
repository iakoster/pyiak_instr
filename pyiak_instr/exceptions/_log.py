from ._base import PyiError


__all__ = [
    "CompletedWorkError",
    "InterruptedWorkError"
]


class CompletedWorkError(PyiError):
    """
    Raised when the work function has already been
    called before and cannot be called again
    """

    def __init__(self, work_name: str):
        PyiError.__init__(
            self,
            "Work %s is already done" % work_name,
            work_name
        )
        self.work_name = work_name


class InterruptedWorkError(PyiError):
    """
    Raised when the interruption reason is indicated and
    an attempt is made to change the steps or call the work
    """

    def __init__(self, reason):
        PyiError.__init__(
            self,
            "Work was interrupted by %r" % reason,
            reason
        )
        self.reason = reason

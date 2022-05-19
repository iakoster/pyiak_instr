from ._base import PyiError


__all__ = [
    "FloatWordsCountError",
    "PartialFieldError"
]


class FloatWordsCountError(PyiError):

    def __init__(self, field: str, expected: int, got: int | float):
        PyiError.__init__(
            self,
            f"not integer count of words in the {field}. "
            f"Expected {expected}, got {got:.1f}"
        )
        self.field = field
        self.expected = expected
        self.got = got
        self.args = (self.message, self.field, self.expected, self.got)


class PartialFieldError(PyiError):

    def __init__(self, field: str, occup: int | float):
        PyiError.__init__(
            self,
            f"The {field} is incomplete. The field is filled to {occup:.1f}"
        )
        self.field = field
        self.occupancy = occup
        self.args = (self.message, self.field, self.occupancy)

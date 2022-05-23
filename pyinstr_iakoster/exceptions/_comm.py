from ._base import PyiError


__all__ = [
    "FieldError",
    "FloatWordsCountError",
    "PartialFieldError"
]


class FieldError(PyiError):

    def __init__(self, msg: str, field: str, *args):
        PyiError.__init__(self, msg)
        self.field = field
        self.args = (self.message, field, *args)


class FloatWordsCountError(FieldError):

    def __init__(self, field: str, expected: int, got: int | float):
        FieldError.__init__(
            self,
            f"not integer count of words in the {field}. "
            f"Expected {expected}, got {got:.1f}",
            field, expected, got
        )
        self.expected = expected
        self.got = got


class PartialFieldError(FieldError):

    def __init__(self, field: str, occup: int | float):
        FieldError.__init__(
            self,
            f"The {field} is incomplete. The field is filled to {occup:.1f}",
            field, occup
        )
        self.occupancy = occup

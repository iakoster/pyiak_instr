from ._base import PyiError


__all__ = [
    "MessageError",
    "NotConfiguredMessageError",
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
            f"not integer count of words in the {field} "
            f"(expected {expected}, got {got:.1f})",
            field, expected, got
        )
        self.expected = expected
        self.got = got


class PartialFieldError(FieldError):

    def __init__(self, field: str, occup: int | float):
        FieldError.__init__(
            self,
            f"the {field} is incomplete (filled to {occup:.1f})",
            field, occup
        )
        self.occupancy = occup


class MessageError(PyiError):

    def __init__(self, msg: str, message: str, *args):
        PyiError.__init__(self, msg)
        self.message_class = message
        self.args = (self.message, message, *args)


class MessageContentError(MessageError):

    def __init__(self, message: str, field: str):
        MessageError.__init__(
            self,
            "Content in %s field of %s message is incorrect" % (
                message, field
            ),
            message, field
        )


class NotConfiguredMessageError(MessageError):

    def __init__(self, message: str):
        MessageError.__init__(
            self,
            "fields in %s instanse not configured" % message,
            message
        )

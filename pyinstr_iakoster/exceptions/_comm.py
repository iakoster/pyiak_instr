from ._base import PyiError


__all__ = [
    "MessageError",
    "MessageContentError",
    "NotConfiguredMessageError",
    "FieldError",
    "FloatWordsCountError",
    "PartialFieldError"
]


class FieldError(PyiError):

    def __init__(self, msg: str, field: str, *args):
        PyiError.__init__(self, msg, field, *args)
        self.field = field


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

    def __init__(self, field: str, occupancy: int | float):
        FieldError.__init__(
            self,
            f"the {field} is incomplete (filled to {occupancy:.1f})",
            field, occupancy
        )
        self.occupancy = occupancy


class MessageError(PyiError):

    def __init__(self, msg: str, message: str, *args):
        PyiError.__init__(self, msg, message, *args)
        self.message_class = message


class MessageContentError(MessageError):

    def __init__(self, message: str, field: str, clarification: str = None):
        msg = "Error with %s in %s" % (field, message)
        if clarification is not None:
            msg += ": " + clarification
        MessageError.__init__(self, msg, message, field)


class NotConfiguredMessageError(MessageError):

    def __init__(self, message: str):
        MessageError.__init__(
            self,
            "fields in %s instanse not configured" % message,
            message
        )

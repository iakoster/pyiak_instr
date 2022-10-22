from ._base import PyiError


__all__ = [
    "MessageError",
    "MessageContentError",
    "NotConfiguredMessageError",
    "FieldError",
    "FieldContentError",
]


class FieldError(PyiError):  # nodesc

    def __init__(self, msg: str, field: object, *args):
        PyiError.__init__(self, msg, field, *args)
        self.field = field


class FieldContentError(FieldError):  # nodesc

    def __init__(self, field: type, *args, clarification: str = None):
        msg = "invalid content in %s" % field.__name__
        if clarification is not None:
            msg += ": " + clarification
        FieldError.__init__(self, msg, field, *args)


class MessageError(PyiError):  # nodesc

    def __init__(self, msg: str, message: str, *args):
        PyiError.__init__(self, msg, message, *args)
        self.message_class = message


class MessageContentError(MessageError):  # nodesc

    def __init__(self, message: str, field: str, clarification: str = None):
        msg = "Error with %s in %s" % (field, message)
        if clarification is not None:
            msg += ": " + clarification
        MessageError.__init__(self, msg, message, field)


class NotConfiguredMessageError(MessageError):  # nodesc

    def __init__(self, message: str):
        MessageError.__init__(
            self,
            "fields in %s instanse not configured" % message,
            message
        )

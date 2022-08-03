from ._base import (PyiError)
from ._log import (
    CompletedWorkError,
    InterruptedWorkError,
)
from ._rwfile import (
    FilepathPatternError
)
from ._comm import (
    MessageError,
    MessageContentError,
    NotConfiguredMessageError,
    FieldError,
    FloatWordsCountError,
    PartialFieldError,
)
# todo: simplify exceptions

__all__ = [
    "CompletedWorkError",
    "FieldError",
    "FilepathPatternError",
    "FloatWordsCountError",
    "InterruptedWorkError",
    "MessageContentError",
    "MessageError",
    "NotConfiguredMessageError",
    "PartialFieldError",
    "PyiError",
]

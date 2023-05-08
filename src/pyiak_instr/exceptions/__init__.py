"""
===============================
Exceptions (:mod:`pyiak_instr`)
===============================
"""
from ._base import PyiError
from ._common import (
    CodeNotAllowed,
    NotConfiguredYet,
    NotAmongTheOptions,
    NotSupportedMethod,
    WithoutParent,
)
from ._communication import ContentError
from ._rwfile import RWFileError, FileSuffixError


__all__ = [
    "CodeNotAllowed",
    "ContentError",
    "FileSuffixError",
    "NotAmongTheOptions",
    "NotConfiguredYet",
    "NotSupportedMethod",
    "PyiError",
    "RWFileError",
    "WithoutParent",
]


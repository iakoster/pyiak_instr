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
    WithoutParent,
)
from ._rwfile import RWFileError, FileSuffixError


__all__ = [
    "PyiError",
    "CodeNotAllowed",
    "NotConfiguredYet",
    "NotAmongTheOptions",
    "WithoutParent",
    "RWFileError",
    "FileSuffixError",
]

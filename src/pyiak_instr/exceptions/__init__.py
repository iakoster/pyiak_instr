"""
===============================
Exceptions (:mod:`pyiak_instr`)
===============================
"""
from ._base import PyiError
from ._common import CodeNotAllowed, NotConfiguredYet, WithoutParent
from ._rwfile import RWFileError, FileSuffixError


__all__ = [
    "PyiError",
    "CodeNotAllowed",
    "NotConfiguredYet",
    "WithoutParent",
    "RWFileError",
    "FileSuffixError",
]

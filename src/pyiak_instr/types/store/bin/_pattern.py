from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from abc import abstractmethod
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Protocol,
    Self,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

from ....core import Code
# from ....rwfile import RWConfig
from ....exceptions import NotConfiguredYet
from ....typing import WithBaseStringMethods
from ..._pattern import (
    MetaPatternABC,
    PatternABC,
    WritablePatternABC,
)
from ._bin import BytesStorageABC
from ._struct import (
    BytesDecodeT,
    BytesEncodeT,
    BytesFieldStructABC,
    BytesStorageStructABC,
)


__all__ = [
    "BytesFieldStructPatternABC",
    "BytesStorageStructPatternABC",
    "BytesStoragePatternABC",
]


FieldStructT = TypeVar("FieldStructT", bound=BytesFieldStructABC)
StorageStructT = TypeVar(
    "StorageStructT", bound=BytesStorageStructABC[BytesFieldStructABC]
)
StorageT = TypeVar("StorageT", bound=BytesStorageABC)

FieldStructPatternT = TypeVar(
    "FieldStructPatternT", bound="BytesFieldStructPatternABC"
)
StorageStructPatternT = TypeVar(
    "StorageStructPatternT", bound="BytesStorageStructPatternABC"
)
StoragePatternT = TypeVar(
    "StoragePatternT", bound="BytesStoragePatternABC"
)


class BytesFieldStructPatternABC(PatternABC[FieldStructT]):
    """
    Represent abstract class of pattern for bytes struct (field).
    """

    _required_init_parameters = {"bytes_expected"}

    @property
    def is_dynamic(self) -> bool:
        """
        Returns
        -------
        bool
            True if the pattern can be interpreted as a dynamic,
            otherwise - False.
        """
        return self.size <= 0

    @property
    def size(self) -> int:
        """
        Returns
        -------
        int
            size of the field in bytes.
        """
        return cast(int, self._kw["bytes_expected"])


class BytesStorageStructPatternABC(
    MetaPatternABC[StorageStructT, FieldStructPatternT]
):
    """
    Represent abstract class of pattern for bytes struct (storage).
    """

    _sub_p_par_name = "fields"

    def _modify_each(
        self,
        changes_allowed: bool,
        name: str,
        for_sub: dict[str, Any],
    ) -> dict[str, Any]:
        for_sub = super()._modify_each(changes_allowed, name, for_sub)
        for_sub["name"] = name
        return for_sub


class BytesStoragePatternABC(MetaPatternABC[StorageT, StorageStructPatternT]):

    _sub_p_par_name = "storage"

    def __init__(self, typename: str, name: str, **kwargs: Any):
        super().__init__(typename, name, pattern=self, **kwargs)

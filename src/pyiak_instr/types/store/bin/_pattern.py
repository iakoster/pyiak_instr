from __future__ import annotations
from pathlib import Path
from configparser import ConfigParser
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
from ..._rwdata import RWData
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
        if "bytes_expected" in self._kw:
            return cast(int, self._kw["bytes_expected"])

        start = self._kw["start"] if "start" in self._kw else 0
        stop = self._kw["stop"] if "stop" in self._kw else None

        if stop is None:
            if start < 0:
                return -start
        elif start >= 0 and stop > 0 or start < 0 and stop < 0:
            return stop - start
        return 0


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


class BytesStoragePatternABC(
    MetaPatternABC[StorageT, StorageStructPatternT], WritablePatternABC
):

    _rwdata: type[RWData[ConfigParser]]
    _sub_p_par_name = "storage"

    def configure(self, **patterns: StorageStructPatternT) -> Self:
        if len(patterns) > 1:
            raise TypeError(
                f"only one storage pattern allowed, got {len(patterns)}"
            )
        return super().configure(**patterns)

    def write(self, path: Path) -> None:
        """
        Write pattern configuration to config file.

        Parameters
        ----------
        path : Path
            path to config file.

        Raises
        ------
        NotConfiguredYet
            is patterns is not configured yet.
        """
        if len(self._sub_p) == 0:
            raise NotConfiguredYet(self)

        (name, pattern), = self._sub_p.items()
        pars = {
            "_": self.__init_kwargs__(),
            name: pattern.__init_kwargs__(),
            # todo: access to sub-patterns in MetaPattern
            **{n: p.__init_kwargs__() for n, p in pattern._sub_p.items()},
        }

        with self._rwdata(path) as cfg:
            if cfg.api.has_section(name):
                cfg.api.remove_section(name)
            cfg.set({name: pars})
            cfg.commit()

    @classmethod
    def read(cls, path: Path, *keys: str) -> Self:
        """
        Read init kwargs from `path` and initialize class instance.

        Parameters
        ----------
        path : Path
            path to the file.
        *keys : str
            keys to search required pattern in file. Must include only one
            argument - `name`.

        Returns
        -------
        Self
            initialized self instance.

        Raises
        ------
        TypeError
            if given invalid count of keys.
        """
        if len(keys) != 1:
            raise TypeError(
                f"{cls.__name__} takes only 1 argument "
                f"({len(keys)} given)"
            )
        (name,) = keys

        with cls._rwdata(path) as cfg:
            opts = cfg.api.options(name)
            opts.pop(opts.index("_"))
            opts.pop(opts.index(name))

            # todo: access to sub-pattern type in MetaPattern
            field_type = cls._sub_p_type._sub_p_type
            return cls(**cfg.get(name, "_")).configure(
                **{name: cls._sub_p_type(**cfg.get(name, name)).configure(
                    **{f: field_type(**cfg.get(name, f)) for f in opts}
                )}
            )

    def _get_parameters_dict(
        self,
        changes_allowed: bool,
        additions: dict[str, Any],
    ) -> dict[str, Any]:
        parameters = super()._get_parameters_dict(changes_allowed, additions)

        storage, = parameters[self._sub_p_par_name].values()
        parameters["storage"] = storage

        parameters["pattern"] = self
        return parameters

    def _modify_each(
            self,
            changes_allowed: bool,
            name: str,
            for_sub: dict[str, Any],
    ) -> dict[str, Any]:
        for_sub = super()._modify_each(changes_allowed, name, for_sub)
        for_sub["name"] = name
        return for_sub

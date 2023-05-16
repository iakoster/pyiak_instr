"""Private module of ``pyiak_instr.types.store`` with types for store
module."""
# pylint: disable=too-many-lines
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
from ._struct import (
    BytesDecodeT,
    BytesEncodeT,
    BytesFieldStructABC,
    BytesStorageStructABC,
)


__all__ = [
    # "BytesFieldABC",
    # "BytesFieldPatternABC",
    "BytesStorageABC",
    # "BytesStoragePatternABC",
    # "ContinuousBytesStoragePatternABC",
]

FieldStructT = TypeVar("FieldStructT", bound=BytesFieldStructABC)
StorageStructT = TypeVar(
    "StorageStructT", bound=BytesStorageStructABC[BytesFieldStructABC]
)


class BytesStorageABC(
    WithBaseStringMethods, Generic[FieldStructT, StorageStructT]
):
    """
    Represents abstract class for bytes storage.

    Parameters
    ----------
    storage : StorageStructT
        storage struct instance.
    """

    def __init__(self, storage: StorageStructT) -> None:
        self._s = storage
        self._c = bytearray()

    @overload
    def decode(self, field: str) -> BytesDecodeT:
        ...

    @overload
    def decode(self) -> dict[str, BytesDecodeT]:
        ...

    def decode(self, *args: str) -> BytesDecodeT | dict[str, BytesDecodeT]:
        """
        Decode content.

        Parameters
        ----------
        *args : str
            arguments for function
        """
        args_len = len(args)
        if args_len == 0:
            return self._s.decode(self._c)
        if args_len == 1 and isinstance(args[0], str):
            return self._s.decode(args[0], self.content(args[0]))
        raise TypeError("invalid arguments")

    @overload
    def encode(self, content: bytes) -> Self:
        ...

    @overload
    def encode(self, **fields: BytesEncodeT) -> Self:
        ...

    def encode(self, *args: bytes, **kwargs: BytesEncodeT) -> Self:
        is_empty = self.is_empty()
        encoded = self._s.encode(*args, all_fields=is_empty, **kwargs)

        if is_empty:
            for content in encoded.values():
                self._c += content

        else:
            for name, content in encoded.items():
                self._c[self._s[name].slice_] = content

        return self

    # kinda properties

    def bytes_count(self, field: str | None = None):
        return len(self.content(field))

    def content(self, field: str | None = None) -> bytes:
        """
        Returns
        -------
        bytes
            content of the storage.
        """
        content = self._c if field is None else self._c[self._s[field].slice_]
        return bytes(content)

    def is_empty(self, field: str | None = None) -> bool:
        return self.bytes_count(field) == 0

    @overload
    def words_count(self) -> dict[str, int]:
        ...

    @overload
    def words_count(self, field: str) -> int:
        ...

    def words_count(self, *args: str) -> int | dict[str, int]:
        if len(args) == 0:
            return {
                n: self.bytes_count(n) // f.words_bytesize
                for n, f in self._s.items()
            }

        if len(args) == 1:
            name, = args
            return self.bytes_count(name) // self._s[name].words_bytesize

        raise TypeError("invalid arguments")

    @property
    def struct(self) -> StorageStructT:
        """
        Returns
        -------
        StorageStructT
            storage struct.
        """
        return self._s

    def __str_field_content__(self, name: str) -> str:
        field = self._s[name]
        content = self.content(name)
        length = len(content)
        if length == 0:
            return "EMPTY"

        step = field.word_bytesize
        if length > 20 and self.words_count(name) > 2:
            if step == 1:
                border = 4
            elif step == 2:
                border = 6
            elif step in {3, 4}:
                border = 2 * step
            else:
                border = step
        else:
            border = length

        string, start = "", 0
        while start < length:
            if start == border:
                start = length - border
                string += "... "

            stop = start + step
            word = content[start:stop].hex().lstrip("0")
            string += (word.upper() if len(word) else "0") + (
                " " if stop != length else ""
            )
            start = stop

        return string

    def __str_under_brackets__(self) -> str:
        if self.is_empty():
            return "EMPTY"
        return ", ".join(
           f"{n}={self.__str_field_content__(n)}" for n, _ in self._s.items()
        )

    def __bytes__(self) -> bytes:
        return self.content()

    def __len__(self) -> int:
        """Bytes count in message"""
        return self.bytes_count()


# # todo: verify current content
# # todo: __format__
# class BytesFieldABC(WithBaseStringMethods, Generic[StorageT, StructT]):
#     """
#     Represents base parser class for work with field content.
#
#     Parameters
#     ----------
#     storage: StorageT
#         bytes storage instance
#     name : str
#         field name.
#     struct : StructT
#         field structure instance.
#     """
#
#     def __init__(self, storage: StorageT, name: str, struct: StructT) -> None:
#         self._storage = storage
#         self._name = name
#         self._struct = struct
#
#
#     @overload
#     def __getitem__(self, index: int) -> int | float:
#         ...
#
#     @overload
#     def __getitem__(self, index: slice) -> npt.NDArray[np.int_ | np.float_]:
#         ...
#
#     def __getitem__(
#         self, index: int | slice
#     ) -> int | float | npt.NDArray[np.int_ | np.float_]:
#         """
#         Parameters
#         ----------
#         index : int | slice
#             word index.
#
#         Returns
#         -------
#         int | float | NDArray[int | float]
#             word value.
#         """
#         return self.decode()[index]
#
#
# # todo: create storage struct (dataclass)
# # todo: __format__
# class BytesStorageABC(
#     WithBaseStringMethods, Generic[ParentPatternT, ParserT, StructT]
# ):
#     """
#     Represents abstract class for bytes storage.
#
#     Parameters
#     ----------
#     name : str
#         name of storage configuration.
#     fields : dict[str, StructT]
#         dictionary of fields.
#     pattern : ParentPatternT | None, default=None
#         storage pattern.
#     """
#
#     _struct_field: dict[type[StructT], type[ParserT]]
#
#     @property
#     def has_pattern(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             if True - storage has parent pattern.
#         """
#         return self._p is not None
#
#     @property
#     def pattern(self) -> ParentPatternT:
#         """
#         Returns
#         -------
#         ParentPatternT
#             parent pattern of self instance.
#
#         Raises
#         ------
#         AttributeError
#             if parent pattern is None (not set).
#         """
#         if not self.has_pattern:
#             raise AttributeError(
#                 f"'{self.__class__.__name__}' object has no parent pattern"
#             )
#         return cast(ParentPatternT, self._p)
#
#
# class BytesFieldPatternABC(PatternABC[StructT], Generic[StructT]):
#     """
#     Represent abstract class of pattern for bytes struct (field).
#     """
#
#     _required_init_parameters = {"bytes_expected"}
#
#     @property
#     def is_dynamic(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True if the pattern can be interpreted as a dynamic,
#             otherwise - False.
#         """
#         return self.size <= 0
#
#     @property
#     def size(self) -> int:
#         """
#         Returns
#         -------
#         int
#             size of the field in bytes.
#         """
#         return cast(int, self._kw["bytes_expected"])
#
#
# class BytesStoragePatternABC(
#     MetaPatternABC[StorageT, PatternT],
#     WritablePatternABC,
#     Generic[StorageT, PatternT],
# ):
#     """
#     Represent abstract class of pattern for bytes storage.
#
#     Parameters
#     ----------
#     typename: str
#         name of pattern target type.
#     name: str
#         name of pattern meta-object format.
#     **kwargs: Any
#         parameters for target initialization.
#     """
#
#     _sub_p_par_name = "fields"
#
#     def __init__(self, typename: str, name: str, **kwargs: Any):
#         super().__init__(typename, name, pattern=self, **kwargs)
#
#     def write(self, path: Path) -> None:
#         """
#         Write pattern configuration to config file.
#
#         Parameters
#         ----------
#         path : Path
#             path to config file.
#
#         Raises
#         ------
#         NotConfiguredYet
#             is patterns is not configured yet.
#         """
#         # if len(self._sub_p) == 0:
#         #     raise NotConfiguredYet(self)
#         # pars = {
#         #     self._name: self.__init_kwargs__(),
#         #     **{n: p.__init_kwargs__() for n, p in self._sub_p.items()},
#         # }
#         #
#         # with RWConfig(path) as cfg:
#         #     if cfg.api.has_section(self._name):
#         #         cfg.api.remove_section(self._name)
#         #     cfg.set({self._name: pars})
#         #     cfg.commit()
#
#     @classmethod
#     def read(cls, path: Path, *keys: str) -> Self:
#         """
#         Read init kwargs from `path` and initialize class instance.
#
#         Parameters
#         ----------
#         path : Path
#             path to the file.
#         *keys : str
#             keys to search required pattern in file. Must include only one
#             argument - `name`.
#
#         Returns
#         -------
#         Self
#             initialized self instance.
#
#         Raises
#         ------
#         TypeError
#             if given invalid count of keys.
#         """
#         # if len(keys) != 1:
#         #     raise TypeError(f"given {len(keys)} keys, expect one")
#         # (name,) = keys
#         #
#         # with RWConfig(path) as cfg:
#         #     opts = cfg.api.options(name)
#         #     opts.pop(opts.index(name))
#         #     return cls(**cfg.get(name, name)).configure(
#         #         **{f: cls._sub_p_type(**cfg.get(name, f)) for f in opts}
#         #     )
#
#     def __init_kwargs__(self) -> dict[str, Any]:
#         init_kw = super().__init_kwargs__()
#         init_kw.pop("pattern")
#         return init_kw
#
#
# class ContinuousBytesStoragePatternABC(
#     BytesStoragePatternABC[StorageT, PatternT],
#     Generic[StorageT, PatternT],
# ):
#     """
#     Represents methods for configure continuous storage.
#
#     It's means `start` of the field is equal to `stop` of previous field
#     (e.g. without gaps in content).
#     """
#
#     def _modify_all(
#         self, changes_allowed: bool, for_subs: dict[str, dict[str, Any]]
#     ) -> dict[str, dict[str, Any]]:
#         """
#         Modify additional kwargs for sub-pattern objects.
#
#         Parameters
#         ----------
#         changes_allowed : bool
#             if True allows situations where keys from the pattern overlap
#             with kwargs.
#         for_subs : dict[str, dict[str, Any]]
#             additional kwargs for sub-pattern object if format
#             {FIELD: {PARAMETER: VALUE}}.
#
#         Returns
#         -------
#         dict[str, dict[str, Any]]
#             modified additional kwargs for sub-pattern object.
#         """
#         for_subs = super()._modify_all(changes_allowed, for_subs)
#         dyn_name = self._modify_before_dyn(for_subs)
#         if dyn_name is not None:
#             self._modify_after_dyn(dyn_name, for_subs)
#         return for_subs
#
#     def _modify_before_dyn(
#         self, for_subs: dict[str, dict[str, Any]]
#     ) -> str | None:
#         """
#         Modify `for_subs` up to dynamic field.
#
#         Parameters
#         ----------
#         for_subs : dict[str, dict[str, Any]]
#             additional kwargs for sub-pattern object if format
#             {FIELD: {PARAMETER: VALUE}}.
#
#         Returns
#         -------
#         str | None
#             name of the dynamic field. If None - there is no dynamic field.
#         """
#         start = 0
#         for (name, pattern), kw in zip(
#             self._sub_p.items(), for_subs.values()
#         ):
#             if pattern.is_dynamic:
#                 kw.update(start=start)
#                 return name
#
#             kw.update(start=start)
#             start += pattern.size
#         return None
#
#     def _modify_after_dyn(
#         self,
#         dyn_name: str,
#         for_subs: dict[str, dict[str, Any]],
#     ) -> None:
#         """
#         Modify `for_subs` from dynamic field to end.
#
#         Parameters
#         ----------
#         dyn_name : str
#             name of the dynamic field.
#         for_subs : dict[str, dict[str, Any]]
#             additional kwargs for sub-pattern object if format
#             {FIELD: {PARAMETER: VALUE}}.
#
#         Raises
#         ------
#         TypeError
#             if there is tow dynamic fields.
#         AssertionError
#             if for some reason the dynamic field is not found.
#         """
#         start = 0
#         for name in list(self._sub_p)[::-1]:
#             pattern, kw = self._sub_p[name], for_subs[name]
#
#             if pattern.is_dynamic:
#                 if name == dyn_name:
#                     kw.update(stop=start if start != 0 else None)
#                     return
#                 raise TypeError("two dynamic field not allowed")
#
#             start -= pattern.size
#             kw.update(start=start)
#
#         raise AssertionError("dynamic field not found")

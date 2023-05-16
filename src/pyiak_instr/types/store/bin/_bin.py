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
from ._struct import BytesFieldStructABC, BytesStorageStructABC


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

StructT = TypeVar("StructT", bound="BytesFieldStructProtocol")
ParserT = TypeVar("ParserT", bound="BytesFieldABC[Any, Any]")
StorageT = TypeVar("StorageT", bound="BytesStorageABC[Any, Any, Any]")
PatternT = TypeVar("PatternT", bound="BytesFieldPatternABC[Any]")
ParentPatternT = TypeVar(
    "ParentPatternT", bound="BytesStoragePatternABC[Any, Any]"
)


class BytesStorageABC(
    WithBaseStringMethods, Generic[FieldStructT, StorageStructT]
):
    """
    Represents abstract class for bytes storage.

    Parameters
    ----------
    fields : dict[str, StructT]
        dictionary of fields.
    pattern : ParentPatternT | None, default=None
        storage pattern.
    **storage_kw : Any
        keyword arguments for storage struct.
    """

    _storage_struct_type: type[StorageStructT]

    def __init__(
            self, fields: dict[str, FieldStructT], **storage_kw: Any
    ) -> None:
        if len(fields) == 0:
            raise ValueError(f"{self.__class__.__name__} without fields")

        self._s = self._storage_struct_type(fields=fields, **storage_kw)
        self._c = bytearray()

    @overload
    def decode(self, field: str) -> npt.NDArray[np.int_, np.float_]:
        ...

    @overload
    def decode(self) -> dict[str, npt.NDArray[np.int_, np.float_]]:
        ...

    def decode(
            self, *args: str
    ) -> dict[str, npt.NDArray[np.int_, np.float_]]:
        """
        Decode content.

        Parameters
        ----------
        *args : str
            arguments for function
        """
        match args:
            case (str() as name,):
                return self._s.decode(name, self.content(name))

            case tuple():
                return self._s.decode(self._c)

            case _:
                raise TypeError("invalid arguments")

    @overload
    def content(self, field: str) -> bytes:
        ...

    @overload
    def content(self) -> bytes:
        ...

    def content(self, *args: str) -> bytes:
        """
        Returns
        -------
        bytes
            content of the storage.
        """
        match args:
            case (str() as name,):
                return bytes(self._c[self._s[name].slice_])

            case tuple():
                return bytes(self._c)

            case _:
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

    def __str_under_brackets__(self) -> str:
        return "EMPTY"
        # if len(self._c) == 0 and len(self._f) > 1:
        #     return "EMPTY"
        # return ", ".join(
        #     map(lambda x: f"{x.name}={x.__str_under_brackets__()}", self)
        # )

    def __len__(self) -> int:
        """Bytes count in message"""
        return len(self._c)


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
#     def encode(self, content: int | float | Iterable[int | float]) -> None:
#         """
#         Encode content to bytes and set .
#
#         Parameters
#         ----------
#         content : int | float | Iterable[int | float]
#             content to encoding.
#         """
#         self._storage.change(self._name, content)
#
#     def verify(self, content: bytes) -> bool:
#         """
#         Check the content for compliance with the field parameters.
#
#         Parameters
#         ----------
#         content: bytes
#             content for validating.
#
#         Returns
#         -------
#         bool
#             True - content is correct, False - not.
#         """
#         return self._struct.verify(content)
#
#     @property
#     def bytes_count(self) -> int:
#         """
#         Returns
#         -------
#         int
#             bytes count of the content.
#         """
#         return len(self.content)
#
#     @property
#     def content(self) -> bytes:
#         """
#         Returns
#         -------
#         bytes
#             field content.
#         """
#         return self._storage.content[self._struct.slice_]
#
#     @property
#     def is_empty(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True - if field content is empty.
#         """
#         return len(self.content) == 0
#
#     @property
#     def name(self) -> str:
#         """
#         Returns
#         -------
#         str
#             field name.
#         """
#         return self._name
#
#     @property
#     def struct(self) -> StructT:
#         """
#         Returns
#         -------
#         StructT
#             struct instance.
#         """
#         return self._struct
#
#     @property
#     def words_count(self) -> int:
#         """
#         Returns
#         -------
#         int
#             count of words in the field.
#         """
#         return self.bytes_count // self._struct.word_bytesize
#
#     def __str_under_brackets__(self) -> str:
#         content = self.content
#         length = len(content)
#         if length == 0:
#             return "EMPTY"
#
#         step = self.struct.word_bytesize
#         if length > 20 and self.words_count > 2:
#             if step == 1:
#                 border = 4
#             elif step == 2:
#                 border = 6
#             elif step in {3, 4}:
#                 border = 2 * step
#             else:
#                 border = step
#         else:
#             border = length
#
#         string, start = "", 0
#         while start < length:
#             if start == border:
#                 start = length - border
#                 string += "... "
#
#             stop = start + step
#             word = content[start:stop].hex().lstrip("0")
#             string += (word.upper() if len(word) else "0") + (
#                 " " if stop != length else ""
#             )
#             start = stop
#
#         return string
#
#     def __bytes__(self) -> bytes:
#         """
#         Returns
#         -------
#         bytes
#             field content.
#         """
#         return self.content
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
#     def __iter__(self) -> Iterator[int | float]:
#         """
#         Yields
#         ------
#         int | float
#             word value.
#         """
#         return (el for el in self.decode())
#
#     def __len__(self) -> int:
#         """
#         Returns
#         -------
#         int
#             bytes count of the content.
#         """
#         return self.bytes_count
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
#     def __init__(
#         self,
#         name: str,
#         fields: dict[str, StructT],
#         pattern: ParentPatternT | None = None,
#     ) -> None:
#         if len(fields) == 0:
#             raise ValueError(f"{self.__class__.__name__} without fields")
#
#         self._name = name
#         self._f = fields
#         self._p = pattern
#         self._c = bytearray()
#
#         for f_name, struct in self._f.items():
#             if struct.is_dynamic:
#                 self._dyn_field = f_name
#                 break
#         else:
#             self._dyn_field = ""
#
#     def change(
#         self, name: str, content: int | float | Iterable[int | float]
#     ) -> None:
#         """
#         Change content of one field by name.
#
#         Parameters
#         ----------
#         name : str
#             field name.
#         content : bytes
#             new field content.
#
#         Raises
#         ------
#         TypeError
#             if the message is empty.
#         """
#         if len(self) == 0:
#             raise TypeError("message is empty")
#         parser = self[name]
#         self._c[parser.struct.slice_] = self._encode_content(parser, content)
#
#     def decode(self) -> dict[str, npt.NDArray[np.int_ | np.float_]]:
#         """
#         Iterate by fields end decode each.
#
#         Returns
#         -------
#         dict[str, npt.NDArray[Any]]
#             dictionary with decoded content where key is a field name.
#         """
#         return {n: f.decode() for n, f in self.items()}
#
#     @overload
#     def encode(self, content: bytes) -> Self:
#         ...
#
#     @overload
#     def encode(self, **fields: int | float | Iterable[int | float]) -> Self:
#         ...
#
#     def encode(  # type: ignore[misc]
#         self,
#         content: bytes = b"",
#         **fields: int | float | Iterable[int | float],
#     ) -> Self:
#         """
#         Encode new content to storage.
#
#         Parameters
#         ----------
#         content : bytes, default=b''
#             new full content for storage.
#         **fields : int | float | Iterable[int | float]
#             content for each field.
#
#         Returns
#         -------
#         Self
#             self instance.
#
#         Raises
#         ------
#         TypeError
#             if trying to set full content and content for each field;
#             if full message or fields list is empty.
#         """
#         if len(content) != 0 and len(fields) != 0:
#             raise TypeError("takes a message or fields (both given)")
#
#         if len(content) != 0:
#             self._extract(content)
#         elif len(fields) != 0:
#             self._set(fields)
#         else:
#             raise TypeError("message is empty")
#
#         return self
#
#     def _set(
#         self, fields: dict[str, int | float | Iterable[int | float]]
#     ) -> None:
#         """
#         Set fields content.
#
#         Parameters
#         ----------
#         fields : dict[str, int | float | Iterable[int | float]]
#             dictionary of fields content where key is a field name.
#         """
#         if len(self) == 0:
#             self._set_all(fields)
#         else:
#             for name, content in fields.items():
#                 self.change(name, content)
#
#
#     @staticmethod
#     def _encode_content(
#         parser: ParserT,
#         raw: Iterable[int | float] | int | float,
#     ) -> bytes:
#         """
#         Get new content to the field.
#
#         Parameters
#         ----------
#         parser: str
#             field parser.
#         raw: ArrayLike
#             new content.
#
#         Returns
#         -------
#         bytes
#             new field content
#
#         Raises
#         ------
#         ValueError
#             if new content is not correct for field.
#         """
#         if isinstance(raw, bytes):
#             content = raw
#         else:
#             content = parser.struct.encode(raw)  # todo: bytes support
#
#         if not parser.verify(content):
#             raise ValueError(
#                 f"'{content.hex(' ')}' is not correct for '{parser.name}'"
#             )
#
#         return content
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

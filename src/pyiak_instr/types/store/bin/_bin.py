"""Private module of ``pyiak_instr.types.store`` with types for store
module."""
# pylint: disable=too-many-lines
from typing import (
    TYPE_CHECKING,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from ....typing import WithBaseStringMethods

from ._struct import (
    BytesDecodeT,
    BytesEncodeT,
    BytesFieldStructABC,
    BytesStorageStructABC,
)

if TYPE_CHECKING:
    from ._pattern import BytesStoragePatternABC


__all__ = ["BytesStorageABC"]

FieldStructT = TypeVar("FieldStructT", bound=BytesFieldStructABC)
StorageStructT = TypeVar(
    "StorageStructT", bound=BytesStorageStructABC[BytesFieldStructABC]
)
StoragePatternT = TypeVar("StoragePatternT", bound="BytesStoragePatternABC")


# todo: verify current content
# todo: __format__
class BytesStorageABC(
    WithBaseStringMethods,
    Generic[FieldStructT, StorageStructT, StoragePatternT],
):
    """
    Represents abstract class for bytes storage.

    Parameters
    ----------
    storage : StorageStructT
        storage struct instance.
    """

    def __init__(
            self,
            storage: StorageStructT,
            pattern: StoragePatternT | None = None,
    ) -> None:
        self._s = storage
        self._p = pattern
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
        match args:
            case _ if len(args) == 0:
                return self._s.decode(self.content())

            case (str() as name,):
                return self._s.decode(name, self.content(name))

            case _:
                raise TypeError(f"invalid arguments")

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
        match args:
            case _ if len(args) == 0:
                return {
                    n: self.bytes_count(n) // f.word_bytesize
                    for n, f in self._s.items()
                }

            case (str() as name,):
                return self.bytes_count(name) // self._s[name].word_bytesize

            case _:
                raise TypeError("invalid arguments")

    @property
    def has_pattern(self) -> bool:
        """
        Returns
        -------
        bool
            if True - storage has parent pattern.
        """
        return self._p is not None

    @property
    def pattern(self) -> StoragePatternT:
        """
        Returns
        -------
        StoragePatternT
            parent pattern of self instance.

        Raises
        ------
        AttributeError
            if parent pattern is None (not set).
        """
        if not self.has_pattern:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no parent pattern"
            )
        return cast(StoragePatternT, self._p)

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

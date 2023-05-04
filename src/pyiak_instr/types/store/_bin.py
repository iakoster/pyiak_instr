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

from ...core import Code
from ...rwfile import RWConfig
from ...exceptions import NotConfiguredYet
from ...typing import WithBaseStringMethods
from .._pattern import MetaPatternABC, PatternABC, WritablePatternABC


__all__ = [
    "STRUCT_DATACLASS",
    "BytesFieldABC",
    "BytesFieldPatternABC",
    "BytesFieldStructProtocol",
    "BytesStorageABC",
    "BytesStoragePatternABC",
    "ContinuousBytesStoragePatternABC",
]


StructT = TypeVar("StructT", bound="BytesFieldStructProtocol")
ParserT = TypeVar("ParserT", bound="BytesFieldABC[Any, Any]")
StorageT = TypeVar("StorageT", bound="BytesStorageABC[Any, Any, Any]")
PatternT = TypeVar("PatternT", bound="BytesFieldPatternABC[Any]")
ParentPatternT = TypeVar(
    "ParentPatternT", bound="BytesStoragePatternABC[Any, Any]"
)


STRUCT_DATACLASS = dataclass(frozen=True, kw_only=True)


@STRUCT_DATACLASS
class BytesFieldStructProtocol(Protocol):
    """
    Represents protocol for field structure.
    """

    start: int = 0
    """the number of bytes in the message from which the fields begin."""

    stop: int | None = None
    """index of stop byte. If None - stop is end of bytes."""

    bytes_expected: int = 0
    """expected bytes count for field. If less than 1, from the start byte
    to the end of the message."""

    fmt: Code = Code.U8
    """format for packing or unpacking the content.
    The word length is calculated from the format."""

    order: Code = Code.BIG_ENDIAN
    """bytes order for packing and unpacking."""

    default: bytes = b""  # todo: to ContentType
    """default value of the field."""

    def __post_init__(self) -> None:
        if self.stop == 0:
            raise ValueError("'stop' can't be equal to zero")
        if self.stop is not None and self.bytes_expected > 0:
            raise TypeError("'bytes_expected' and 'stop' setting not allowed")
        if 0 > self.start > -self.bytes_expected:
            raise ValueError("it will be out of bounds")

        self._modify_values()

        if self.bytes_expected % self.word_bytesize:
            raise ValueError(
                "'bytes_expected' does not match an integer word count"
            )

    @abstractmethod
    def decode(self, content: bytes) -> npt.NDArray[np.int_ | np.float_]:
        """
        Decode content from bytes with parameters from struct.

        Parameters
        ----------
        content : bytes
            content for decoding.

        Returns
        -------
        npt.NDArray[np.int_ | np.float_]
            decoded content.
        """

    @abstractmethod
    def encode(self, content: int | float | Iterable[int | float]) -> bytes:
        """
        Encode content to bytes with parameters from struct.

        Parameters
        ----------
        content : int | float | Iterable[int | float]
            content for encoding.

        Returns
        -------
        bytes
            encoded content.
        """

    @property
    @abstractmethod
    def word_bytesize(self) -> int:
        """
        Returns
        -------
        int
            count of bytes in one word.
        """

    # todo: return Code and if raise=True - raise ContentError
    def verify(self, content: bytes) -> bool:
        """
        Verify that `content` is correct for the given field structure.

        Parameters
        ----------
        content : bytes
            content to verifying.

        Returns
        -------
        bool
            True - content is correct, False - is not.
        """
        if self.is_dynamic:
            return len(content) % self.word_bytesize == 0
        return len(content) == self.bytes_expected

    def _modify_values(self) -> None:
        """
        Modify values of the struct.

        Raises
        ------
        AssertionError
            if in some reason `start`, `stop` or `bytes_expected` can't be
            modified.
        """
        if self.bytes_expected < 0:
            object.__setattr__(self, "bytes_expected", 0)

        if self.bytes_expected > 0:
            stop = self.start + self.bytes_expected
            if stop != 0:
                object.__setattr__(self, "stop", stop)

        elif self.stop is not None:
            if not self.start >= 0 > self.stop:
                object.__setattr__(
                    self, "bytes_expected", self.stop - self.start
                )

        elif self.start <= 0 and self.stop is None:
            object.__setattr__(self, "bytes_expected", -self.start)

        elif not self.is_dynamic:
            raise AssertionError(
                "impossible to modify start, stop and bytes_expected"
            )

    @property
    def has_default(self) -> bool:
        """
        Returns
        -------
        bool
            True - default more than zero.
        """
        return len(self.default) != 0

    @property
    def is_dynamic(self) -> bool:
        """
        Returns
        -------
        bool
            if True - field is dynamic (from empty to any).
        """
        return self.bytes_expected == 0

    @property
    def slice_(self) -> slice:
        """
        Returns
        -------
        slice
            slice with start and stop indexes of field.
        """
        return slice(self.start, self.stop)

    @property
    def words_expected(self) -> int:
        """
        Returns
        -------
        int
            expected words count in the field. Returns 0 if field is infinite.
        """
        return self.bytes_expected // self.word_bytesize


# todo: verify current content
# todo: __format__
class BytesFieldABC(WithBaseStringMethods, Generic[StorageT, StructT]):
    """
    Represents base parser class for work with field content.

    Parameters
    ----------
    storage: StorageT
        bytes storage instance
    name : str
        field name.
    struct : StructT
        field structure instance.
    """

    def __init__(self, storage: StorageT, name: str, struct: StructT) -> None:
        self._storage = storage
        self._name = name
        self._struct = struct

    def decode(self) -> npt.NDArray[np.int_ | np.float_]:
        """
        Decode field content.

        Returns
        -------
        NDArray
            decoded content.
        """
        return self._struct.decode(self.content)

    def encode(self, content: int | float | Iterable[int | float]) -> None:
        """
        Encode content to bytes and set .

        Parameters
        ----------
        content : int | float | Iterable[int | float]
            content to encoding.
        """
        self._storage.change(self._name, content)

    def verify(self, content: bytes) -> bool:
        """
        Check the content for compliance with the field parameters.

        Parameters
        ----------
        content: bytes
            content for validating.

        Returns
        -------
        bool
            True - content is correct, False - not.
        """
        return self._struct.verify(content)

    @property
    def bytes_count(self) -> int:
        """
        Returns
        -------
        int
            bytes count of the content.
        """
        return len(self.content)

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self._storage.content[self._struct.slice_]

    @property
    def is_empty(self) -> bool:
        """
        Returns
        -------
        bool
            True - if field content is empty.
        """
        return len(self.content) == 0

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            field name.
        """
        return self._name

    @property
    def struct(self) -> StructT:
        """
        Returns
        -------
        StructT
            struct instance.
        """
        return self._struct

    @property
    def words_count(self) -> int:
        """
        Returns
        -------
        int
            count of words in the field.
        """
        return self.bytes_count // self._struct.word_bytesize

    def __str_under_brackets__(self) -> str:
        content = self.content
        length = len(content)
        if length == 0:
            return "EMPTY"

        step = self.struct.word_bytesize
        if length > 20 and self.words_count > 2:
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

    def __bytes__(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self.content

    @overload
    def __getitem__(self, index: int) -> int | float:
        ...

    @overload
    def __getitem__(self, index: slice) -> npt.NDArray[np.int_ | np.float_]:
        ...

    def __getitem__(
        self, index: int | slice
    ) -> int | float | npt.NDArray[np.int_ | np.float_]:
        """
        Parameters
        ----------
        index : int | slice
            word index.

        Returns
        -------
        int | float | NDArray[int | float]
            word value.
        """
        return self.decode()[index]

    def __iter__(self) -> Iterator[int | float]:
        """
        Yields
        ------
        int | float
            word value.
        """
        return (el for el in self.decode())

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            bytes count of the content.
        """
        return self.bytes_count


# todo: create storage struct (dataclass)
# todo: __format__
class BytesStorageABC(
    WithBaseStringMethods, Generic[ParentPatternT, ParserT, StructT]
):
    """
    Represents abstract class for bytes storage.

    Parameters
    ----------
    name : str
        name of storage configuration.
    fields : dict[str, StructT]
        dictionary of fields.
    pattern : ParentPatternT | None, default=None
        storage pattern.
    """

    _struct_field: dict[type[StructT], type[ParserT]]

    def __init__(
        self,
        name: str,
        fields: dict[str, StructT],
        pattern: ParentPatternT | None = None,
    ) -> None:
        if len(fields) == 0:
            raise ValueError(f"{self.__class__.__name__} without fields")

        self._name = name
        self._f = fields
        self._p = pattern
        self._c = bytearray()

    def change(
        self, name: str, content: int | float | Iterable[int | float]
    ) -> None:
        """
        Change content of one field by name.

        Parameters
        ----------
        name : str
            field name.
        content : bytes
            new field content.

        Raises
        ------
        TypeError
            if the message is empty.
        """
        if len(self) == 0:
            raise TypeError("message is empty")
        parser = self[name]
        self._c[parser.struct.slice_] = self._encode_content(parser, content)

    def decode(self) -> dict[str, npt.NDArray[np.int_ | np.float_]]:
        """
        Iterate by fields end decode each.

        Returns
        -------
        dict[str, npt.NDArray[Any]]
            dictionary with decoded content where key is a field name.
        """
        return {n: f.decode() for n, f in self.items()}

    @overload
    def encode(self, content: bytes) -> Self:
        ...

    @overload
    def encode(self, **fields: int | float | Iterable[int | float]) -> Self:
        ...

    def encode(  # type: ignore[misc]
        self,
        content: bytes = b"",
        **fields: int | float | Iterable[int | float],
    ) -> Self:
        """
        Encode new content to storage.

        Parameters
        ----------
        content : bytes, default=b''
            new full content for storage.
        **fields : int | float | Iterable[int | float]
            content for each field.

        Returns
        -------
        Self
            self instance.

        Raises
        ------
        TypeError
            if trying to set full content and content for each field;
            if full message or fields list is empty.
        """
        if len(content) != 0 and len(fields) != 0:
            raise TypeError("takes a message or fields (both given)")

        if len(content) != 0:
            self._extract(content)
        elif len(fields) != 0:
            self._set(fields)
        else:
            raise TypeError("message is empty")

        return self

    def items(self) -> Iterator[tuple[str, ParserT]]:
        """
        Returns
        -------
        Iterable[tuple[str, ParserT]]
            Iterable of names and parsers.
        """
        return ((f.name, f) for f in self)

    def _check_fields_list(self, fields: set[str]) -> None:
        """
        Check that fields names is correct.

        Parameters
        ----------
        fields : set[str]
            set of field names for setting.

        Raises
        ------
        AttributeError
            if extra or missing field names founded.
        """
        diff = set(self._f).symmetric_difference(fields)
        for name in diff.copy():
            if name in self:
                parser = self[name]
                if (
                    parser.struct.has_default
                    or parser.struct.is_dynamic
                    or len(parser) != 0
                ):
                    diff.remove(name)

        if len(diff) != 0:
            raise AttributeError(
                "missing or extra fields were found: "
                f"{', '.join(map(repr, sorted(diff)))}"
            )

    def _extract(self, content: bytes) -> None:
        """
        Extract fields from existing bytes content.

        Parameters
        ----------
        content: bytes
            new content.

        Raises
        ------
        ValueError
            if content length smaller than minimal storage length
            (`bytes_expected`).
        """
        minimum_size = self.minimum_size
        if len(content) < minimum_size:
            raise ValueError("bytes content too short")
        if not self.is_dynamic and len(content) > minimum_size:
            raise ValueError("bytes content too long")

        if len(self) != 0:
            self._c = bytearray()
        self._set_all({p.name: content[p.struct.slice_] for p in self})

    def _set(
        self, fields: dict[str, int | float | Iterable[int | float]]
    ) -> None:
        """
        Set fields content.

        Parameters
        ----------
        fields : dict[str, int | float | Iterable[int | float]]
            dictionary of fields content where key is a field name.
        """
        if len(self) == 0:
            self._set_all(fields)
        else:
            for name, content in fields.items():
                self.change(name, content)

    def _set_all(
        self, fields: dict[str, int | float | Iterable[int | float]]
    ) -> None:
        """
        Set content to empty field.

        Parameters
        ----------
        fields : dict[str, int | float | Iterable[int | float]]
            dictionary of fields content where key is a field name.

        Raises
        ------
        AssertionError
            if in some reason message is not empty.
        """
        assert len(self) == 0, "message must be empty"

        self._check_fields_list(set(fields))
        for name, parser in self.items():
            if name in fields:
                self._c += self._encode_content(parser, fields[name])

            elif parser.struct.has_default:
                self._c += parser.struct.default

            elif parser.struct.is_dynamic:
                continue

            else:
                raise AssertionError(
                    f"it is impossible to set the value of the '{name}' field"
                )

    @staticmethod
    def _encode_content(
        parser: ParserT,
        raw: Iterable[int | float] | int | float,
    ) -> bytes:
        """
        Get new content to the field.

        Parameters
        ----------
        parser: str
            field parser.
        raw: ArrayLike
            new content.

        Returns
        -------
        bytes
            new field content

        Raises
        ------
        ValueError
            if new content is not correct for field.
        """
        if isinstance(raw, bytes):
            content = raw
        else:
            content = parser.struct.encode(raw)  # todo: bytes support

        if not parser.verify(content):
            raise ValueError(
                f"'{content.hex(' ')}' is not correct for '{parser.name}'"
            )

        return content

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            content of the storage.
        """
        return bytes(self._c)

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
    def is_dynamic(self) -> bool:
        """
        Returns
        -------
        bool
            True - at least one field is dynamic.
        """
        return any(p.struct.is_dynamic for p in self)

    @property
    def minimum_size(self) -> int:
        """
        Returns
        -------
        int
            minimum message size in bytes.
        """
        return sum(s.bytes_expected for s in self._f.values())

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            name of the storage.
        """
        return self._name

    @property
    def pattern(self) -> ParentPatternT:
        """
        Returns
        -------
        ParentPatternT
            parent pattern of self instance.

        Raises
        ------
        AttributeError
            if parent pattern is None (not set).
        """
        if self._p is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no parent pattern"
            )
        return self._p

    def __str_under_brackets__(self) -> str:
        if len(self._c) == 0 and len(self._f) > 1:
            return "EMPTY"
        return ", ".join(
            map(lambda x: f"{x.name}={x.__str_under_brackets__()}", self)
        )

    def __contains__(self, name: str) -> bool:
        """Check that field name in message."""
        return name in self._f

    def __getitem__(self, name: str) -> ParserT:
        """Get field parser."""
        struct = self._f[name]
        return self._struct_field[struct.__class__](self, name, struct)

    def __iter__(self) -> Iterator[ParserT]:
        """Iterate by field parsers."""
        return (self[n] for n in self._f)

    def __len__(self) -> int:
        """Bytes count in message"""
        return len(self._c)


class BytesFieldPatternABC(PatternABC[StructT], Generic[StructT]):
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


class BytesStoragePatternABC(
    MetaPatternABC[StorageT, PatternT],
    WritablePatternABC,
    Generic[StorageT, PatternT],
):
    """
    Represent abstract class of pattern for bytes storage.

    Parameters
    ----------
    typename: str
        name of pattern target type.
    name: str
        name of pattern meta-object format.
    **kwargs: Any
        parameters for target initialization.
    """

    _sub_p_par_name = "fields"

    def __init__(self, typename: str, name: str, **kwargs: Any):
        super().__init__(typename, name, pattern=self, **kwargs)

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
        pars = {
            self._name: self.__init_kwargs__(),
            **{n: p.__init_kwargs__() for n, p in self._sub_p.items()},
        }

        with RWConfig(path) as cfg:
            if cfg.api.has_section(self._name):
                cfg.api.remove_section(self._name)
            cfg.set({self._name: pars})
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
            raise TypeError(f"given {len(keys)} keys, expect one")
        (name,) = keys

        with RWConfig(path) as cfg:
            opts = cfg.api.options(name)
            opts.pop(opts.index(name))
            return cls(**cfg.get(name, name)).configure(
                **{f: cls._sub_p_type(**cfg.get(name, f)) for f in opts}
            )

    def __init_kwargs__(self) -> dict[str, Any]:
        init_kw = super().__init_kwargs__()
        init_kw.pop("pattern")
        return init_kw


class ContinuousBytesStoragePatternABC(
    BytesStoragePatternABC[StorageT, PatternT],
    Generic[StorageT, PatternT],
):
    """
    Represents methods for configure continuous storage.

    It's means `start` of the field is equal to `stop` of previous field
    (e.g. without gaps in content).
    """

    def _modify_all(
        self, changes_allowed: bool, for_subs: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Modify additional kwargs for sub-pattern objects.

        Parameters
        ----------
        changes_allowed : bool
            if True allows situations where keys from the pattern overlap
            with kwargs.
        for_subs : dict[str, dict[str, Any]]
            additional kwargs for sub-pattern object if format
            {FIELD: {PARAMETER: VALUE}}.

        Returns
        -------
        dict[str, dict[str, Any]]
            modified additional kwargs for sub-pattern object.
        """
        for_subs = super()._modify_all(changes_allowed, for_subs)
        dyn_name = self._modify_before_dyn(for_subs)
        if dyn_name is not None:
            self._modify_after_dyn(dyn_name, for_subs)
        return for_subs

    def _modify_before_dyn(
        self, for_subs: dict[str, dict[str, Any]]
    ) -> str | None:
        """
        Modify `for_subs` up to dynamic field.

        Parameters
        ----------
        for_subs : dict[str, dict[str, Any]]
            additional kwargs for sub-pattern object if format
            {FIELD: {PARAMETER: VALUE}}.

        Returns
        -------
        str | None
            name of the dynamic field. If None - there is no dynamic field.
        """
        start = 0
        for (name, pattern), kw in zip(
            self._sub_p.items(), for_subs.values()
        ):
            if pattern.is_dynamic:
                kw.update(start=start)
                return name

            kw.update(start=start)
            start += pattern.size
        return None

    def _modify_after_dyn(
        self,
        dyn_name: str,
        for_subs: dict[str, dict[str, Any]],
    ) -> None:
        """
        Modify `for_subs` from dynamic field to end.

        Parameters
        ----------
        dyn_name : str
            name of the dynamic field.
        for_subs : dict[str, dict[str, Any]]
            additional kwargs for sub-pattern object if format
            {FIELD: {PARAMETER: VALUE}}.

        Raises
        ------
        TypeError
            if there is tow dynamic fields.
        AssertionError
            if for some reason the dynamic field is not found.
        """
        start = 0
        for name in list(self._sub_p)[::-1]:
            pattern, kw = self._sub_p[name], for_subs[name]

            if pattern.is_dynamic:
                if name == dyn_name:
                    kw.update(stop=start if start != 0 else None)
                    return
                raise TypeError("two dynamic field not allowed")

            start -= pattern.size
            kw.update(start=start)

        raise AssertionError("dynamic field not found")

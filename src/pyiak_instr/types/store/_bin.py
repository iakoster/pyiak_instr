"""Private module of ``pyiak_instr.types.store`` with types for store
module."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    Generator,
    Iterable,
    Protocol,
    Self,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

from .._pattern import MetaPatternABC, PatternABC, WritablePatternABC
from ...rwfile import RWConfig
from ...exceptions import NotConfiguredYet
from ...typing import SupportsContainsGetitem


__all__ = [
    "BytesFieldABC",
    "BytesFieldStructProtocol",
    "BytesStorageABC",
    "BytesStoragePatternABC",
    "ContinuousBytesStoragePatternABC",
]


StructT = TypeVar("StructT", bound="BytesFieldStructProtocol")
ParserT = TypeVar("ParserT", bound="BytesFieldABC[Any]")
StorageT = TypeVar("StorageT", bound="BytesStorageABC[Any, Any]")
PatternT = TypeVar("PatternT", bound=PatternABC[Any])


@dataclass(frozen=True, kw_only=True)
class BytesFieldStructProtocol(Protocol):
    """
    Represents protocol for field structure.
    """

    start: int
    """the number of bytes in the message from which the fields begin."""

    stop: int | None
    """index of stop byte. If None - stop is end of bytes."""

    bytes_expected: int
    """expected bytes count for field. If less than 1, from the start byte
    to the end of the message."""

    default: bytes  # todo: to ContentType
    """default value of the field."""

    def __post_init__(self) -> None:
        if self.stop == 0:
            raise ValueError("'stop' can't be equal to zero")
        if self.stop is not None and self.bytes_expected > 0:
            raise TypeError("'bytes_expected' or 'stop' setting allowed")

        if self.bytes_expected < 0:
            object.__setattr__(self, "bytes_expected", 0)

        if self.stop is not None:
            if (
                self.start >= 0
                and self.stop > 0
                or self.start < 0
                and self.stop < 0
            ):
                object.__setattr__(
                    self, "bytes_expected", self.stop - self.start
                )

        elif self.bytes_expected > 0:
            stop = self.start + self.bytes_expected
            if stop != 0:
                object.__setattr__(self, "stop", stop)

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

    @abstractmethod
    def validate(self, content: bytes) -> bool:  # todo: to verify.
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

    @property
    @abstractmethod
    def word_bytesize(self) -> int:
        """
        Returns
        -------
        int
            count of bytes in one word.
        """

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
    def is_floating(self) -> bool:
        """
        Returns
        -------
        bool
            if True - field is floating (from empty to any).
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


class BytesFieldABC(ABC, Generic[StructT]):
    """
    Represents base parser class for work with field content.

    Parameters
    ----------
    name : str
        field name.
    struct : StructT
        field structure instance.
    """

    def __init__(self, name: str, struct: StructT) -> None:
        self._name = name
        self._struct = struct

    @property
    @abstractmethod
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """

    def decode(self) -> npt.NDArray[np.int_ | np.float_]:
        """
        Decode field content.

        Returns
        -------
        NDArray
            decoded content.
        """
        return self._struct.decode(self.content)

    def encode(self, content: int | float | Iterable[int | float]) -> bytes:
        """
        Encode content to bytes.

        Parameters
        ----------
        content : int | float | Iterable[int | float]
            content to encoding.

        Returns
        -------
        bytes
            encoded bytes.
        """
        return self._struct.encode(content)

    def validate(self, content: bytes) -> bool:
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
        return self._struct.validate(content)

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

    def __bytes__(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self.content

    def __getitem__(
        self, index: int | slice
    ) -> np.int_ | np.float_ | npt.NDArray[np.int_ | np.float_]:
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

    def __iter__(self) -> Generator[int | float, None, None]:
        """
        Yields
        ------
        int | float
            word value.
        """
        for item in self.decode():
            yield item

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            bytes count of the content.
        """
        return self.bytes_count


class BytesStorageABC(ABC, Generic[ParserT, StructT]):
    """
    Represents abstract class for bytes storage.
    """

    def __init__(self, name: str, fields: dict[str, StructT]) -> None:
        self._name = name
        self._f = fields
        self._c = bytearray()

    @abstractmethod
    def __getitem__(self, name: str) -> ParserT:
        """Get field parser."""

    def decode(self) -> dict[str, npt.NDArray[np.int_ | np.float_]]:
        """
        Iterate by fields end decode each.

        Returns
        -------
        dict[str, npt.NDArray[Any]]
            dictionary with decoded content where key is a field name.
        """
        return {p.name: p.decode() for p in self}

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

    def items(self) -> Iterable[tuple[str, ParserT]]:
        """
        Returns
        -------
        Iterable[tuple[str, ParserT]]
            Iterable of names and parsers.
        """
        return ((n, self[n]) for n in self._f)

    def _change_field_content(self, name: str, content: bytes) -> None:
        """
        Change content of one field by name.

        Parameters
        ----------
        name : str
            field name.
        content : bytes
            new field content.
        """
        self._c[self[name].struct.slice_] = content

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
                    or parser.struct.is_floating
                    or len(parser) != 0
                ):
                    diff.remove(name)

        if len(diff) != 0:
            raise AttributeError(
                "missing or extra fields were found: "
                f"{', '.join(map(repr, sorted(diff)))}"
            )

    def _encode_field_content(
        self,
        name: str,
        content: Iterable[int | float] | int | float,
    ) -> bytes:
        """
        Get new content to the field.

        Parameters
        ----------
        name: str
            field name.
        content: ArrayLike
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
        parser = self[name]
        if not isinstance(content, bytes):
            content = parser.encode(content)

        if not parser.validate(content):
            raise ValueError(
                f"'{content.hex(' ')}' is not correct for '{parser.name}'"
            )

        return content

    def _extract(self, content: bytes) -> None:
        """
        Extract fields from existing bytes content.

        Parameters
        ----------
        content: bytes
            new content.
        """
        self._set({p.name: content[p.struct.slice_] for p in self})

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
            self._set_full_content(fields)
        else:
            for name, content in fields.items():
                self._change_field_content(
                    name, self._encode_field_content(name, content)
                )

    def _set_full_content(
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
                content = fields[name]
                if isinstance(content, bytes) and len(content) == 0:
                    continue
                self._c += self._encode_field_content(name, content)

            elif parser.struct.has_default:
                self._c += parser.struct.default

            elif parser.struct.is_floating:
                continue

            else:
                raise AssertionError(
                    f"it is impossible to set the value of the '{name}' field"
                )

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
    def name(self) -> str:
        """
        Returns
        -------
        str
            name of the storage.
        """
        return self._name

    def __contains__(self, name: str) -> bool:
        """Check that field name in message."""
        return name in self._f

    def __iter__(self) -> Generator[ParserT, None, None]:
        """Iterate by field parsers."""
        return (self[n] for n in self._f)

    def __len__(self) -> int:
        """Bytes count in message"""
        return len(self._c)


class BytesStoragePatternABC(
    MetaPatternABC[StorageT, PatternT],
    WritablePatternABC,
    Generic[StorageT, PatternT],
):
    """
    Represent abstract class of bytes storage.
    """

    _only_auto_parameters = {"fields"}

    @abstractmethod
    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> StorageT:
        ...

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

    @staticmethod
    def _is_dynamic_pattern(
        kwargs: dict[str, Any], pattern: PatternT
    ) -> bool:  # todo: tests
        """
        Returns True if the joined parameters can be interpreted as a dynamic.

        Parameters
        ----------
        kwargs : dict[str, Any]
            additional parameters.
        pattern : PatternT
            pattern instance

        Returns
        -------
        bool
            True if the `kwargs` or `pattern` can be interpreted as a
            dynamic, otherwise - False
        """

        def check(
            obj: SupportsContainsGetitem,
        ) -> bool:
            """
            Check that the object can be interpreted as a dynamic.

            Parameters
            ----------
            obj : SupportsContainsGetitem
                object for checking.

            Returns
            -------
            bool
                True if the object can be interpreted as a dynamic,
                otherwise - False
            """
            return (
                "stop" in obj
                and obj["stop"] is None
                or "bytes_expected" in obj
                and isinstance(obj["bytes_expected"], int)
                and obj["bytes_expected"] <= 0
            )

        return check(kwargs) or check(pattern)


class ContinuousBytesStoragePatternABC(
    BytesStoragePatternABC[StorageT, PatternT],
    Generic[StorageT, PatternT, StructT],
):
    """
    Represents methods for configure continuous storage.

    It's means `start` of the field is equal to `stop` of previous field
    (e.g. without gaps in content).
    """

    _only_auto_parameters = {"fields", "start", "stop"}

    @abstractmethod
    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> StorageT:
        ...

    def _get_continuous(
        self,
        changes_allowed: bool,
        for_storage: dict[str, Any],
        for_fields: dict[str, dict[str, Any]],
    ) -> StorageT:
        """
        Get initialized continuous storage.

        Parameters
        ----------
        changes_allowed: bool
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection, otherwise the
            `additions` take precedence.
        for_storage: dict[str, Any]:
            dictionary with parameters for storage in format
            {PARAMETER: VALUE}.
        for_fields: dict[str, dict[str, Any]]
            dictionary with parameters for fields in format
            {FIELD: {PARAMETER: VALUE}}.

        Returns
        -------
        ContinuousBytesStorage
            initialized storage.

        Raises
        ------
        SyntaxError
            if changes are not allowed, but there is an attempt to modify
            the parameter.
        TypeError
            if trying to set 'fields'.
        """
        storage_kw = self._get_parameters_dict(changes_allowed, for_storage)

        fields, dyn_name, dyn_start = self._get_fields_before_dyn(
            changes_allowed, for_fields
        )

        if dyn_start >= 0:
            after, dyn_stop = self._get_fields_after_dyn(
                changes_allowed, for_fields, dyn_name
            )

            dyn_kw = for_fields[dyn_name] if dyn_name in for_fields else {}
            dyn_kw.update(start=dyn_start, stop=dyn_stop)
            fields[dyn_name] = self._sub_p[dyn_name].get(
                changes_allowed=changes_allowed, **dyn_kw
            )

            fields.update(after)

        return self._target(fields=fields, **storage_kw)

    def _get_fields_after_dyn(
        self,
        changes_allowed: bool,
        fields_kw: dict[str, dict[str, Any]],
        dyn: str,
    ) -> tuple[dict[str, StructT], int | None]:
        """
        Get the dictionary of fields that go from infinite field
        (not included) to end.

        Parameters
        ----------
        changes_allowed: bool
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection, otherwise the
            `additions` take precedence.
        fields_kw : dict[str, dict[str, Any]]
            dictionary of kwargs for fields.
        dyn : str
            name of dynamic field.

        Returns
        -------
        tuple[dict[str, OptionsT], str]
            fields - dictionary of fields from infinite (not included);
            dyn_stop - stop index of dynamic field.

        Raises
        ------
        TypeError
            if there is tow dynamic fields.
        AssertionError
            if for some reason the dynamic field is not found.
        """
        start = 0
        fields: list[tuple[str, StructT]] = []
        for name in list(self._sub_p)[::-1]:
            if name == dyn:
                return dict(fields[::-1]), start if start != 0 else None

            pattern = self._sub_p[name]
            field_kw = fields_kw[name] if name in fields_kw else {}
            if self._is_dynamic_pattern(field_kw, pattern):
                raise TypeError("two dynamic field not allowed")

            start -= pattern["bytes_expected"]
            field_kw.update(start=start)
            fields.append(
                (
                    name,
                    pattern.get(changes_allowed=changes_allowed, **field_kw),
                )
            )

        raise AssertionError("dynamic field not found")

    def _get_fields_before_dyn(
        self, changes_allowed: bool, fields_kw: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, StructT], str, int]:
        """
        Get the dictionary of fields that go up to and including the infinite
        field.

        Parameters
        ----------
        changes_allowed: bool
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection, otherwise the
            `additions` take precedence.
        fields_kw : dict[str, dict[str, Any]]
            dictionary of kwargs for fields.

        Returns
        -------
        tuple[dict[str, StructT], str, int]
            fields - dictionary of fields up to infinite (include);
            dyn_name - name of infinite field. Empty if there is no found;
            dyn_start - start index of infinite field. -1 if field is no
                found.
        """
        start: int = 0
        fields: dict[str, StructT] = {}
        for name, pattern in self._sub_p.items():
            field_kw = fields_kw[name] if name in fields_kw else {}
            if self._is_dynamic_pattern(field_kw, pattern):
                return fields, name, start

            field_kw.update(start=start)
            fields[name] = pattern.get(
                changes_allowed=changes_allowed, **field_kw
            )
            start = fields[name].stop  # type: ignore[assignment]

        return fields, "", -1

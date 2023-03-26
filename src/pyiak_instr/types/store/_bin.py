"""Private module of ``pyiak_instr.store`` with types of store module."""
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

from ...utilities import BytesEncoder
from ...rwfile import RWConfig
from ...exceptions import NotConfiguredYet
from ...typing import MetaPatternABC, PatternABC, WritablePatternABC


__all__ = [
    "BytesFieldABC",
    "BytesFieldStructProtocol",
    "BytesStorageABC",
]


StructT = TypeVar("StructT", bound="BytesFieldStructProtocol")
ParserT = TypeVar("ParserT", bound="BytesFieldABC[Any]")
StorageT = TypeVar("StorageT", bound="BytesStorageABC[Any, Any]")
OptionsT = TypeVar("OptionsT")
PatternT = TypeVar("PatternT", bound=PatternABC[Any])


@dataclass(frozen=True, kw_only=True)
class BytesFieldStructProtocol(Protocol):

    start: int

    default: bytes

    _stop: int | None = None

    @abstractmethod
    def decode(self, content: bytes) -> npt.NDArray[np.int_ | np.float_]:
        ...

    @abstractmethod
    def encode(self, content: int | float | Iterable[int | float]) -> bytes:
        ...

    @abstractmethod
    def validate(self, content: bytes) -> bool:
        ...

    @property
    @abstractmethod
    def infinite(self) -> bool:
        ...

    @property
    @abstractmethod
    def slice_(self) -> slice:
        ...

    @property
    @abstractmethod
    def word_length(self) -> int:
        ...

    @property
    def has_default(self) -> bool:
        """
        Returns
        -------
        bool
            True - default more than zero.
        """
        return len(self.default) != 0


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
    def default(self) -> bytes:
        """
        Returns
        -------
        bytes
            default content.
        """
        return self._struct.default

    @property
    def has_default(self) -> bool:
        """
        Returns
        -------
        bool
            True - default more than zero.
        """
        return self._struct.has_default

    @property
    def infinite(self) -> bool:
        """
        Returns
        -------
        bool
            Indicate that it is infinite field.
        """
        return self._struct.infinite

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
    def slice_(self) -> slice:
        """
        Returns
        -------
        slice
            field bytes slice.
        """
        return self._struct.slice_

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
    def words_length(self) -> int:
        """
        Returns
        -------
        int
            count of words in the field.
        """
        return self._struct.word_length

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
        return len(self.content)


class BytesStorageABC(ABC, Generic[ParserT, StructT]):
    def __init__(self, name: str, fields: dict[str, StructT]) -> None:
        self._name = name
        self._f = fields
        self._c = bytearray()

    @abstractmethod
    def __getitem__(self, name: str) -> ParserT:
        """Get field parser."""
        ...

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
        if len(content) != 0 and len(fields) != 0:
            raise TypeError("takes a message or fields (both given)")

        if content is not None:
            self._extract(content)
        elif len(fields) != 0:
            self._set(fields)
        else:
            raise TypeError("message is empty")

        return self

    def items(self) -> Iterable[tuple[str, ParserT]]:
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
        self._c[self[name].slice_] = content

    def _check_fields_list(self, fields: set[str]) -> None:
        diff = set(self._f).symmetric_difference(fields)
        for name in diff:
            if name in self:
                parser = self[name]
                if parser.has_default or parser.infinite or len(parser) != 0:
                    diff.remove(name)

        if len(diff) != 0:
            raise AttributeError(
                "missing or extra fields were found: "
                f"{', '.join(map(repr, diff))}"
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
        self._set({p.name: content[p.slice_] for p in self})

    def _set(
        self, fields: dict[str, int | float | Iterable[int | float]]
    ) -> None:
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
        assert len(self) == 0, "message must be empty"

        self._check_fields_list(set(fields))
        for name, parser in self.items():
            if name in fields:
                content = fields[name]
                if isinstance(content, bytes) and len(content) == 0:
                    continue
                self._c += self._encode_field_content(name, content)

            elif parser.has_default:
                self._c += parser.default

            elif parser.infinite:
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
    MetaPatternABC[OptionsT, PatternT],
    WritablePatternABC,
    Generic[OptionsT, PatternT],
):

    @abstractmethod
    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> OptionsT:
        ...

    def write(self, path: Path) -> None:
        """
        Write pattern configuration to config file.

        Parameters
        ----------
        path : Path
            path to config file.
        """
        if len(self._p) == 0:
            raise NotConfiguredYet(self)
        pars = {
            self._name: self.__init_kwargs__(),
            **{n: p.__init_kwargs__() for n, p in self._p.items()}
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
        name, = keys

        with RWConfig(path) as cfg:
            opts = cfg.api.options(name)
            opts.pop(opts.index(name))
            return cls(**cfg.get(name, name)).configure(
                **{f: cls._sub_pattern(**cfg.get(name, f)) for f in opts}
            )


class ContinuousBytesStoragePatternABC(
    BytesStoragePatternABC[OptionsT, PatternT],
    Generic[OptionsT, PatternT],
):

    def _get_continuous(
        self,
        for_storage: dict[str, Any],
        for_fields: dict[str, dict[str, Any]],
    ) -> PatternT:
        """
        Get initialized continuous storage.

        Parameters
        ----------
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
        """
        storage_kw = self._kw
        if len(for_storage) != 0:
            storage_kw.update(for_storage)

        fields, inf = self._get_fields_before_inf(for_fields)
        if len(fields) != len(self._p):
            after, next_ = self._get_fields_after_inf(for_fields, inf)
            fields.update(after)
            object.__setattr__(fields[inf], "_stop", fields[next_].start)

        # todo: Incompatible return value type (got "OptionsT", expected "PatternT")
        return self._target(**storage_kw, **fields)

    def _get_fields_after_inf(
        self,
        fields_kw: dict[str, dict[str, Any]],
        inf: str,
    ) -> tuple[dict[str, StructT], str]:
        """
        Get the dictionary of fields that go from infinite field
        (not included) to end.

        Parameters
        ----------
        fields_kw : dict[str, dict[str, Any]]
            dictionary of kwargs for fields.
        inf : str
            name of infinite fields.

        Returns
        -------
        tuple[dict[str, BytesFieldStruct], str]
            fields - dictionary of fields from infinite (not included);
            next - name of next field after infinite.
        """
        rev_names = list(self._p)[::-1]
        fields, start, next_ = [], 0, ""
        for name in rev_names:
            if name == inf:
                break

            pattern = self._p[name]
            start -= pattern["expected"] * BytesEncoder.get_bytesize(
                pattern["fmt"]
            )
            field_kw = fields_kw[name] if name in fields_kw else {}
            field_kw.update(start=start)
            fields.append(
                (name, pattern.get(changes_allowed=True, **field_kw))
            )
            next_ = name

        return dict(fields[::-1]), next_

    def _get_fields_before_inf(
        self, fields_kw: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, StructT], str]:
        """
        Get the dictionary of fields that go up to and including the infinite
        field.

        Parameters
        ----------
        fields_kw : dict[str, dict[str, Any]]
            dictionary of kwargs for fields.

        Returns
        -------
        tuple[dict[str, BytesFieldStruct], str]
            fields - dictionary of fields up to infinite (include);
            inf - name of infinite field. Empty if there is no found.
        """
        fields, start, inf = {}, 0, ""
        for name, pattern in self._p.items():
            field_kw = fields_kw[name] if name in fields_kw else {}
            field_kw.update(start=start)

            # not supported, but it is correct behaviour
            # if len(set(for_field).intersection(pattern)) == 0:
            #     fields[name] = pattern.get(**for_field)
            fields[name] = pattern.get(changes_allowed=True, **field_kw)

            stop = fields[name].stop
            if stop is None:
                inf = name
                break
            start = stop

        return fields, inf

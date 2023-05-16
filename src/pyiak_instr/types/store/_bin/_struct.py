"""Private module of ``pyiak_instr.types.store`` with types for store
module."""
from __future__ import annotations
from dataclasses import InitVar, dataclass, field as field_
from abc import ABC
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

from ....core import Code
from ..._encoders import Encoder


__all__ = ["STRUCT_DATACLASS", "BytesFieldStructABC", "BytesStorageStructABC"]


FieldStructT = TypeVar("FieldStructT", bound="BytesFieldStructABC")

BytesDecodeT = npt.NDArray[np.int_ | np.float_]
BytesEncodeT = (
    int | float | bytes | list[int | float] | npt.NDArray[np.int_ | np.float_]
)
EncoderT: TypeAlias = Encoder[BytesDecodeT, BytesEncodeT, bytes]


STRUCT_DATACLASS = dataclass(frozen=True, kw_only=True)


# todo: drop to default
# todo: shortcut decode default?
@STRUCT_DATACLASS
class BytesFieldStructABC(ABC):
    """
    Represents base class for field structure.
    """

    name: str = ""
    """field name."""

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

    encoder: InitVar[
        Callable[[Code, Code], EncoderT], type[EncoderT] | None
    ] = None

    _encoder: EncoderT = field_(default=None, init=False)

    def __post_init__(self, encoder: type[Encoder] | Encoder | None) -> None:
        if self.stop == 0:
            raise ValueError("'stop' can't be equal to zero")
        if self.stop is not None and self.bytes_expected > 0:
            raise TypeError("'bytes_expected' and 'stop' setting not allowed")
        if 0 > self.start > -self.bytes_expected:
            raise ValueError("it will be out of bounds")

        if encoder is None:
            raise ValueError("struct encoder not specified")
        self._set_attr("_encoder", encoder(self.fmt, self.order))
        self._modify_values()

        if self.bytes_expected % self.word_bytesize:
            raise ValueError(
                "'bytes_expected' does not match an integer word count"
            )
        if self.has_default and not self.verify(self.default):
            raise ValueError("default value is incorrect")

    def decode(self, content: bytes) -> BytesDecodeT:
        """
        Decode content from bytes with parameters from struct.

        Parameters
        ----------
        content : bytes
            content for decoding.

        Returns
        -------
        BytesDecodeT
            decoded content.
        """
        return self._encoder.decode(content)

    def encode(self, content: BytesEncodeT) -> bytes:
        """
        Encode content to bytes with parameters from struct.

        Parameters
        ----------
        content : BytesEncodeT
            content for encoding.

        Returns
        -------
        bytes
            encoded content.
        """
        return self._encoder.encode(content)

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
            self._set_attr("bytes_expected", 0)

        if self.bytes_expected > 0:
            stop = self.start + self.bytes_expected
            if stop != 0:
                self._set_attr("stop", stop)

        elif self.stop is not None:
            if not self.start >= 0 > self.stop:
                self._set_attr("bytes_expected", self.stop - self.start)

        elif self.start <= 0 and self.stop is None:
            self._set_attr("bytes_expected", -self.start)

        elif not self.is_dynamic:
            raise AssertionError(
                "impossible to modify start, stop and bytes_expected"
            )

    def _set_attr(self, attr: str, value: Any) -> None:
        object.__setattr__(self, attr, value)

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
    def word_bytesize(self) -> int:
        """
        Returns
        -------
        int
            count of bytes in one word.
        """
        return self._encoder.value_size

    @property
    def words_expected(self) -> int:
        """
        Returns
        -------
        int
            expected words count in the field. Returns 0 if field is infinite.
        """
        return self.bytes_expected // self.word_bytesize


@STRUCT_DATACLASS
class BytesStorageStructABC(ABC, Generic[FieldStructT]):
    """
    Represents base class for storage structure.
    """

    name: str = "std"
    """name of storage configuration."""

    fields: InitVar[dict[str, FieldStructT]] = {}  # type: ignore[assignment]
    """dictionary of fields."""

    dynamic_field_name: str = field_(default="", init=False)

    _f: dict[str, FieldStructT] = field_(default_factory=dict, init=False)

    def __post_init__(self, fields: dict[str, FieldStructT]) -> None:
        if len(fields) == 0:
            raise ValueError(f"{self.__class__.__name__} without fields")
        if "" in fields:
            raise KeyError("empty field name not allowed")
        for name, struct in fields.items():
            s_name = struct.name
            if name != s_name:
                raise KeyError(f"invalid struct name: {name!r} != {s_name!r}")

        object.__setattr__(self, "_f", fields)
        self._modify_values()

    @overload
    def decode(
        self, name: str, content: bytes
    ) -> BytesDecodeT:
        ...

    @overload
    def decode(
        self, content: bytes
    ) -> dict[str, BytesDecodeT]:
        ...

    def decode(
        self, *args: str | bytes, **kwargs: Any
    ) -> (
        npt.NDArray[np.int_ | np.float_]
        | dict[str, npt.NDArray[np.int_ | np.float_]]
    ):
        if len(kwargs):
            raise TypeError("takes no keyword arguments")

        match args:
            case (str() as name, bytes() as content):
                return self[name].decode(content)

            case (bytes() as content,):
                return {f.name: f.decode(content[f.slice_]) for f in self}

            case _:
                raise TypeError("invalid arguments")

    @overload
    def encode(self, content: bytes) -> dict[str, bytes]:
        ...

    @overload
    def encode(self, **fields: BytesEncodeT) -> dict[str, bytes]:
        ...

    def encode(  # type: ignore[misc]
        self, *args: bytes, **fields: BytesEncodeT,
    ) -> dict[str, bytes]:
        if len(args) != 0 and len(fields) != 0:
            raise TypeError("takes a bytes or fields (both given)")
        if len(args) == 0 and len(fields) == 0:
            raise TypeError("missing arguments")

        if len(args) != 0:
            content, = args
            if len(content) == 0:
                raise ValueError("content is empty")
            return {f.name: f.encode(content[f.slice_]) for f in self}

        if len(fields) != 0:
            return {f: self[f].encode(c) for f, c in fields.items()}

        raise AssertionError()

    def items(self) -> Iterator[tuple[str, FieldStructT]]:
        """
        Returns
        -------
        Iterator[tuple[str, ParserT]]
            Iterator of names and parsers.
        """
        return ((f.name, f) for f in self)

    def _modify_values(self) -> None:
        """
        Modify values of the struct.
        """
        for name, struct in self.items():
            if struct.is_dynamic:  # todo: raise if second dynamic
                object.__setattr__(self, "dynamic_field_name", name)
                break

    def _verify_bytes_content(self, content: bytes) -> None:
        """
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

    def _verify_fields_list(self, fields: set[str]) -> None:
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
        diff = self.fields_set.symmetric_difference(fields)
        for name in diff.copy():
            if name in self:
                field = self[name]
                if field.has_default or field.is_dynamic:
                    diff.remove(name)

        if len(diff) != 0:
            raise AttributeError(
                "missing or extra fields were found: "
                f"{', '.join(map(repr, sorted(diff)))}"
            )

    @property
    def fields_set(self) -> set[str]:
        return set(self._f)

    @property
    def is_dynamic(self) -> bool:
        """
        Returns
        -------
        bool
            True - has dynamic field.
        """
        return bool(len(self.dynamic_field_name))

    @property
    def minimum_size(self) -> int:
        """
        Returns
        -------
        int
            minimum message size in bytes.
        """
        # pylint: disable=no-member
        return sum(s.bytes_expected for s in self._f.values())

    def __contains__(self, name: str) -> bool:
        """Check that field name in message."""
        return name in self._f

    def __getitem__(self, name: str) -> FieldStructT:
        """Get field struct."""
        return self._f[name]

    def __iter__(self) -> Generator[FieldStructT, None, None]:
        """Iterate by field structs."""
        for struct in self._f.values():  # pylint: disable=no-member
            yield struct

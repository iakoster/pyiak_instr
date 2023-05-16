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
from ....exceptions import ContentError
from ..._encoders import Encoder


__all__ = [
    "BytesDecodeT",
    "BytesEncodeT",
    "STRUCT_DATACLASS",
    "BytesFieldStructABC",
    "BytesStorageStructABC",
]

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
        Callable[[Code, Code], EncoderT] | type[EncoderT] | None
    ] = None

    _encoder: EncoderT = field_(default=None, init=False)

    def __post_init__(
            self,
            encoder: Callable[[Code, Code], EncoderT] | type[EncoderT] | None,
    ) -> None:
        if self.stop == 0:
            raise ValueError("'stop' can't be equal to zero")
        if self.stop is not None and self.bytes_expected > 0:
            raise TypeError("'bytes_expected' and 'stop' setting not allowed")
        if 0 > self.start > -self.bytes_expected:
            raise ValueError("it will be out of bounds")

        if encoder is None:
            raise ValueError("struct encoder not specified")
        self._setattr("_encoder", encoder(self.fmt, self.order))
        self._modify_values()

        if self.bytes_expected % self.word_bytesize:
            raise ValueError(
                "'bytes_expected' does not match an integer word count"
            )
        if self.has_default and not self.verify(self.default):
            raise ValueError("default value is incorrect")

    def decode(self, content: bytes, verify: bool = False) -> BytesDecodeT:
        """
        Decode content from bytes with parameters from struct.

        Parameters
        ----------
        content : bytes
            content for decoding.
        verify : bool, default=False
            verify `content` before decoding.

        Returns
        -------
        BytesDecodeT
            decoded content.
        """
        if verify:
            self.verify(content, raise_if_false=True)
        return self._encoder.decode(content)

    def encode(self, content: BytesEncodeT, verify: bool = False) -> bytes:
        """
        Encode content to bytes with parameters from struct.

        Parameters
        ----------
        content : BytesEncodeT
            content for encoding.
        verify : bool, default=False
            verify content after encoding.

        Returns
        -------
        bytes
            encoded content.
        """
        encoded = self._encoder.encode(content)
        if verify:
            self.verify(encoded, raise_if_false=True)
        return encoded

    # todo: return Code
    def verify(self, content: bytes, raise_if_false: bool = False) -> bool:
        """
        Verify that `content` is correct for the given field structure.

        Parameters
        ----------
        content : bytes
            content to verifying.
        raise_if_false : bool
            raise `ContentError` if content not correct.

        Returns
        -------
        bool
            True - content is correct, False - is not.
        """
        if self.is_dynamic:
            correct = len(content) % self.word_bytesize == 0
        else:
            correct = len(content) == self.bytes_expected

        if raise_if_false and not correct:
            raise ContentError(self, clarification=content.hex(" "))
        return correct

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
            self._setattr("bytes_expected", 0)

        if self.bytes_expected > 0:
            stop = self.start + self.bytes_expected
            if stop != 0:
                self._setattr("stop", stop)

        elif self.stop is not None:
            if not self.start >= 0 > self.stop:
                self._setattr("bytes_expected", self.stop - self.start)

        elif self.start <= 0 and self.stop is None:
            self._setattr("bytes_expected", -self.start)

        elif not self.is_dynamic:
            raise AssertionError(
                "impossible to modify start, stop and bytes_expected"
            )

    def _setattr(self, attr: str, value: Any) -> None:
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


FieldStructT = TypeVar("FieldStructT", bound=BytesFieldStructABC)


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

        self._setattr("_f", fields)
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
                return self[name].decode(content, verify=True)

            case (bytes() as content,):
                return {
                    f.name: f.decode(
                        content[f.slice_], verify=True
                    ) for f in self
                }

            case _:
                raise TypeError("invalid arguments")

    @overload
    def encode(self, content: bytes) -> dict[str, bytes]:
        ...

    @overload
    def encode(
            self, all_fields: bool = False, **fields: BytesEncodeT
    ) -> dict[str, bytes]:
        ...

    @overload
    def encode(
        self, *args: bytes, all_fields: bool = False, **kwargs: BytesEncodeT,
    ) -> dict[str, bytes]:
        ...

    def encode(
        self, *args: bytes, all_fields: bool = False, **kwargs: BytesEncodeT,
    ) -> dict[str, bytes]:
        if len(args) != 0 and len(kwargs) != 0:
            raise TypeError("takes a bytes or fields (both given)")
        if len(args) == 0 and len(kwargs) == 0:
            raise TypeError("missing arguments")

        match args, kwargs:
            case (bytes() as content,), {} if len(kwargs) == 0:
                self._verify_bytes_content(content)
                return {
                    f.name: f.encode(
                        content[f.slice_], verify=True
                    ) for f in self
                }

            case tuple(), dict() as fields if len(args) == 0:
                if all_fields:
                    return self._get_all_fields(fields)
                return {
                    f: self[f].encode(
                        c, verify=True
                    ) for f, c in fields.items()
                }

            case _:
                raise TypeError("invalid arguments")

    def items(self) -> Generator[tuple[str, FieldStructT], None, None]:
        """
        Yields
        ------
        Generator[tuple[str, FieldStructT], None, None]
            Iterator of names and parsers.
        """
        for field in self:
            yield field.name, field

    def _get_all_fields(
            self, fields: dict[str, BytesEncodeT]
    ) -> dict[str, bytes]:
        """

        Parameters
        ----------
        fields : dict[str, BytesEncodeT]
            dictionary of fields content where key is a field name.

        Raises
        ------
        """
        self._verify_fields_list(set(fields))
        content: dict[str, bytes] = {}

        for name, field in self.items():
            if name in fields:
                content[name] = field.encode(fields[name], verify=True)

            elif field.has_default:
                content[name] = field.default

            elif field.is_dynamic:
                content[name] = b""

            else:
                raise AssertionError(
                    f"it is impossible to encode the value for {name!r}"
                )

        return content

    def _modify_values(self) -> None:
        """
        Modify values of the struct.
        """
        for name, struct in self.items():
            if struct.is_dynamic and self.is_dynamic:
                raise TypeError("two dynamic field not allowed")
            if struct.is_dynamic:
                self._setattr("dynamic_field_name", name)

    def _setattr(self, key: str, value: Any) -> None:
        object.__setattr__(self, key, value)

    def _verify_bytes_content(self, content: bytes) -> None:
        """
        Raises
        ------
        ValueError
            if content length smaller than minimal storage length
            (`bytes_expected`).
        """
        minimum_size, content_len = self.minimum_size, len(content)
        if content_len < minimum_size:
            raise ValueError(
                "bytes content too short: expected at least "
                f"{minimum_size}, got {content_len}"
            )
        if not self.is_dynamic and content_len > minimum_size:
            raise ValueError(
                "bytes content too long: expected "
                f"{minimum_size}, got {content_len}"
            )

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
        diff = set(self._f).symmetric_difference(fields)
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

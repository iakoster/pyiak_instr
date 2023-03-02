"""Private module of ``pyiak_instr.store`` for work with bytes."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Self, Generator

import numpy.typing as npt

from ..utilities import BytesEncoder


__all__ = ["BytesField", "BytesFieldPattern"]


@dataclass(frozen=True, kw_only=True)
class BytesField:
    """
    Represents field parameters with values encoded in bytes.
    """

    start: int
    """the number of bytes in the message from which the fields begin."""

    fmt: str
    """format for packing or unpacking the content.
    The word length is calculated from the format."""

    order: str
    """bytes order for packing and unpacking."""

    expected: int
    """expected number of words in the field. If less than 1, from
    the start byte to the end of the message."""

    default: bytes = b""
    """default value of the field."""

    def __post_init__(self) -> None:
        if self.expected < 0:
            object.__setattr__(self, "expected", 0)

    def decode(self, content: bytes) -> npt.NDArray[Any]:
        """
        Decode content from parent.

        Parameters
        ----------
        content: bytes
            content for decoding.

        Returns
        -------
        npt.NDArray[Any]
            decoded content.
        """
        return BytesEncoder.decode(content, fmt=self.fmt, order=self.order)

    def encode(self, content: npt.ArrayLike) -> bytes:
        """
        Encode array to bytes.

        Parameters
        ----------
        content : npt.ArrayLike
            content to encoding.

        Returns
        -------
        bytes
            encoded content.
        """
        return BytesEncoder.encode(content, fmt=self.fmt, order=self.order)

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
        if self.infinite:
            if len(content) % self.word_size:
                return False
        elif len(content) != self.bytes_expected:
            return False
        return True

    @property
    def bytes_expected(self) -> int:
        """
        Returns
        -------
        int
            expected bytes for field. Returns 0 if field is infinite.
        """
        return self.expected * self.word_size

    @property
    def infinite(self) -> bool:
        """
        Returns
        -------
        bool
            Indicate that it is finite field.
        """
        return not self.expected

    @property
    def slice(self) -> slice:
        """
        Returns
        -------
        slice
            The range of bytes from the message belonging to the field.
        """
        return slice(self.start, self.stop)

    @property
    def stop(self) -> int | None:
        """
        Returns
        -------
        int | None
            The number of byte in the message to which the field stops.
        """
        if self.infinite:
            return None
        return self.start + self.word_size * self.expected

    @property
    def word_size(self) -> int:
        """
        Returns
        -------
        int
            The length of the one word in bytes.
        """
        return struct.calcsize(self.fmt)


# todo: up to this level all functions and properties from BytesField
class BytesFieldParser:
    """
    Represents parser for work with field content.

    Parameters
    ----------
    storage: ContinuousBytesStorage
        storage of fields.
    name: str
        field name.
    field: BytesField
        field instance.
    """

    def __init__(
        self,
        storage: ContinuousBytesStorage,
        name: str,
        field: BytesField,
    ):
        self._name = name
        self._s = storage
        self._f = field

    def decode(self) -> npt.NDArray[Any]:
        """
        Decode field content.

        Returns
        -------
        NDArray
            decoded content.
        """
        return self._f.decode(self.content)

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self._s.content[self._f.slice]

    @property
    def fld(self) -> BytesField:
        """
        Returns
        -------
        BytesField
            field instance.
        """
        return self._f

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            field name
        """
        return self._name

    @property
    def words_count(self) -> int:
        """
        Returns
        -------
        int
            Count of words in the field.
        """
        return len(self) // self._f.word_size

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            bytes count of the content
        """
        return len(self.content)


class ContinuousBytesStorage:
    """
    Represents continuous storage where data storage in bytes.

    Continuous means that the storage fields must go continuously, that is,
    where one field ends, another must begin.

    Parameters
    ----------
    **fields: BytesField
        fields of the storage. The kwarg Key is used as the field name.
    """

    def __init__(self, **fields: BytesField):
        self._f = fields
        self._c = bytearray()

    def extract(self, content: bytes) -> Self:
        """
        Extract fields from existing bytes content.

        Parameters
        ----------
        content: bytes
            new content.

        Returns
        -------
        Self
            self instance.
        """
        fields = {}
        for parser in self:
            new = content[parser.fld.slice]
            if len(new):
                fields[parser.name] = new
        self.set(**fields)
        return self

    def set(self, **fields: npt.ArrayLike) -> Self:
        """
        Set content to the fields.

        Parameters
        ----------
        **fields: ArrayLike
            fields content

        Returns
        -------
        Self
            self instance

        Raises
        ------
        AttributeError
            if values of non-existent fields were passed or values of some
            fields were not passed.
        """

        diff = set()
        raw_diff = set(self._f).symmetric_difference(set(fields))
        for field in raw_diff:
            if field in self and (
                self[field].fld.infinite or len(self[field]) != 0
            ):
                continue
            diff.add(field)

        if len(diff) != 0:
            raise AttributeError(
                "missing or superfluous fields were found: %r" % sorted(diff)
            )

        self._set(fields)
        return self

    def _set(self, fields: dict[str, npt.ArrayLike]) -> None:
        """
        Set new content to the fields.

        Parameters
        ----------
        fields: fields: dict[str, ArrayLike]
            dictionary of new field content.
        """
        for field in self:
            if field.name in fields:
                self._set_field_content(field, fields[field.name])

    def _set_field_content(
        self,
        parser: BytesFieldParser,
        content: npt.ArrayLike,
    ) -> None:
        """
        Set new content to the field.

        Parameters
        ----------
        parser: BytesFieldParser
            field parser.
        content: ArrayLike
            new content.

        Raises
        ------
        ValueError
            if new content is not correct for field.
        """
        new_content = parser.fld.encode(content)
        if not parser.fld.validate(new_content):
            raise ValueError(
                "%r is not correct for %r"
                % (new_content.hex(" "), parser.name)
            )
        self._c[parser.fld.slice] = new_content

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            content of the storage.
        """
        return self._c

    def __contains__(self, name: str) -> bool:
        """Check that field name exists."""
        return name in self._f

    def __getitem__(self, field: str) -> BytesFieldParser:
        """Get field parser."""
        return BytesFieldParser(self, field, self._f[field])

    def __iter__(self) -> Generator[BytesFieldParser, None, None]:
        """Iterate by field parsers."""
        for field in self._f:
            yield self[field]


class BytesFieldPattern:
    """
    Represents class which storage common parameters for field.

    Parameters
    ----------
    **parameters: Any
        common parameters.
    """

    def __init__(self, **parameters: Any):
        self._kw = parameters

    def add(self, key: str, value: Any) -> None:
        """
        Add new parameter to the pattern.

        Parameters
        ----------
        key: str
            new parameter name.
        value: Any
            new parameter value.

        Raises
        ------
        KeyError
            if parameter name is already exists.
        """
        if key in self:
            raise KeyError("parameter in pattern already")
        self._kw[key] = value

    def get(self, **parameters: Any) -> BytesField:
        """
        Get field initialized with parameters from pattern and from
        `parameters`.

        Parameters
        ----------
        **parameters: Any
            additional field initialization parameters.

        Returns
        -------
        BytesField
            initialized field.
        """
        return BytesField(**self._kw, **parameters)

    def get_updated(self, **parameters: Any) -> BytesField:
        """
        Get field initialized with parameters from pattern and from
        `parameters`.

        If parameters from pattern will be updated via `parameters` before
        creation Field instance.

        Parameters
        ----------
        **parameters: Any
            parameters for field.

        Returns
        -------
        BytesField
            initialized field.
        """
        kw_ = self._kw.copy()
        kw_.update(parameters)
        return BytesField(**kw_)  # pylint: disable=missing-kwoa

    def pop(self, key: str) -> Any:
        """
        Extract parameter with removal.

        Parameters
        ----------
        key: str
            parameter name.

        Returns
        -------
        Any
            parameter value.
        """
        return self._kw.pop(key)

    def __contains__(self, item: str) -> bool:
        return item in self._kw

    def __getitem__(self, parameter: str) -> Any:
        return self._kw[parameter]

    def __setitem__(self, parameter: str, value: Any) -> None:
        if parameter not in self:
            raise KeyError("%r not in parameters" % parameter)
        self._kw[parameter] = value

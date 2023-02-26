from __future__ import annotations
import struct
import functools
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt

from ..exceptions import WithoutParent
from ..utilities import BytesEncoder


def _neg_int_to_zero(value: int) -> int:
    """
    Change integer to zero if value less than 1.

    Parameters
    ----------
    value : int
        value.

    Returns
    -------
    int
        changed value.
    """
    if value < 1:
        return 0
    return value


def _parent_dependence(f):  # nodesc
    @functools.wraps(f)
    def wrapper(self: BytesField, *args: Any, **kwargs: Any) -> Any:
        if self.parent is None:
            raise WithoutParent()
        return f(self, *args, **kwargs)

    return wrapper


@dataclass(frozen=True, kw_only=True)
class BytesField(object):
    """
    Represents field parameters with values encoded in bytes.
    """

    start: int
    """the number of bytes in the message from which the fields begin."""

    may_be_empty: bool
    """if True then field can be empty in a message."""

    fmt: str
    """format for packing or unpacking the content. 
    The word length is calculated from the format."""

    order: str
    """bytes order for packing and unpacking."""

    expected: int = field(default_factory=_neg_int_to_zero)
    """expected number of words in the field. If less than 1, from 
    the start byte to the end of the message."""

    default: bytes = b""
    """default value of the field."""

    parent: ContinuousBytesStorage | None = None
    """parent storage."""

    @_parent_dependence
    def decode(self) -> npt.NDArray[Any]:
        """
        Decode content from parent.

        Returns
        -------
        npt.NDArray[Any]
            decoded content.
        """
        return BytesEncoder.decode(
            self.content, fmt=self.fmt, order=self.order
        )

    def encode(self, content: npt.ArrayLike) -> bytes:
        """
        Encode array to bytes.

        Parameters
        ----------
        content : npt.ArrayLike
            content to encoding.

        Returns
        -------
        encoded content.
        """
        return BytesEncoder.encode(content, fmt=self.fmt, order=self.order)

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
    @_parent_dependence
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self.parent.content[self.slice]

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

    @property
    @_parent_dependence
    def words_count(self) -> int:
        """
        Returns
        -------
        int
            Count of words in the field.
        """
        return len(self.content) // self.word_size


class BytesFieldParametersContainer(object):  # nodesc
    def __init__(self, **parameters: Any):
        self._kw = parameters

    def get(self, **parameters: Any) -> BytesField:
        return BytesField(**self._kw, **parameters)

    def __getitem__(self, parameter: str) -> Any:
        return self._kw[parameter]

    def __setitem__(self, parameter: str, value: Any) -> None:
        if parameter not in self._kw:
            raise KeyError("%r not in parameters list" % parameter)
        self._kw[parameter] = value


class ContinuousBytesStorage(object):
    def __init__(self, **fields: BytesField):
        ...

    @property
    def content(self) -> bytes:
        ...


class BytesStoragePattern(object):
    ...

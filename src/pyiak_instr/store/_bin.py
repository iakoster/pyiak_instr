"""Private module of ``pyiak_instr.store`` for work with bytes."""
from __future__ import annotations
import struct
import functools
from dataclasses import dataclass
from typing import Any, Callable

import numpy.typing as npt

from ..exceptions import WithoutParent
from ..utilities import BytesEncoder


__all__ = ["BytesField", "BytesFieldPattern"]


def _parent_dependence(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    @functools.wraps(func)
    def wrapper(self: BytesField) -> Any:
        if self.parent is None:
            raise WithoutParent()
        return func(self)

    return wrapper


@dataclass(frozen=True, kw_only=True)
class BytesField:
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

    expected: int
    """expected number of words in the field. If less than 1, from
    the start byte to the end of the message."""

    default: bytes = b""
    """default value of the field."""

    parent: ContinuousBytesStorage | None = None
    """parent storage."""

    def __post_init__(self) -> None:
        if self.expected < 0:
            object.__setattr__(self, "expected", 0)

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
        return self.parent.content[self.slice]  # type: ignore

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


# pylint: disable=too-few-public-methods
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
        self._c = b""

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            content of the storage.
        """
        return self._c

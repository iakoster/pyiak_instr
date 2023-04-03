"""Private module of ``pyiak_instr.store`` for work with bytes."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as _field
from typing import (
    Any,
    ClassVar,
    Iterable,
)

import numpy as np
import numpy.typing as npt

from ..core import Code
from ..utilities import BytesEncoder, split_complex_dict
from ..exceptions import NotConfiguredYet
from ..types import PatternABC
from ..types.store import (
    BytesFieldABC,
    BytesFieldStructProtocol,
    BytesStorageABC,
    ContinuousBytesStoragePatternABC,
)


__all__ = [
    "BytesField",
    "BytesFieldStruct",
    "ContinuousBytesStorage",
    "BytesFieldPattern",
    "BytesStoragePattern",
]


@dataclass(frozen=True, kw_only=True)
class BytesFieldStruct(BytesFieldStructProtocol):
    """
    Represents field parameters with values encoded in bytes.
    """

    start: int
    """the number of bytes in the message from which the fields begin."""

    bytes_expected: int
    """expected bytes count for field. If less than 1, from the start byte 
    to the end of the message."""

    fmt: Code
    """format for packing or unpacking the content.
    The word length is calculated from the format."""

    order: Code = Code.BIG_ENDIAN
    """bytes order for packing and unpacking."""

    default: bytes = b""  # todo: to ContentType
    """default value of the field."""

    _stop: int | None = _field(default=None, repr=False)
    """stop byte index of field content."""

    ORDERS: ClassVar[dict[Code, str]] = BytesEncoder.ORDERS
    ALLOWED_CODES: ClassVar[set[Code]] = BytesEncoder.ALLOWED_CODES

    def __post_init__(self) -> None:
        BytesEncoder.check_fmt_order(self.fmt, self.order)

        if self.bytes_expected < 0:
            object.__setattr__(self, "bytes_expected", 0)
        if self.bytes_expected % self.word_length:
            raise ValueError(
                "'bytes_expected' does not match an integer word count"
            )

        stop = self.start + self.bytes_expected
        if not self.infinite and stop != 0:
            object.__setattr__(self, "_stop", stop)

    def decode(self, content: bytes) -> npt.NDArray[np.int_ | np.float_]:
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

    def encode(self, content: Iterable[int | float] | int | float) -> bytes:
        """
        Encode content to bytes.

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
            return not(len(content) % self.word_length)
        return len(content) == self.bytes_expected

    @property
    def infinite(self) -> bool:
        """
        Returns
        -------
        bool
            Indicate that it is finite field.
        """
        return not self.bytes_expected

    @property
    def slice_(self) -> slice:
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
        return self._stop

    @property
    def word_length(self) -> int:
        """
        Returns
        -------
        int
            The length of the one word in bytes.
        """
        return BytesEncoder.get_bytesize(self.fmt)

    @property
    def words_expected(self) -> int:
        """
        Returns
        -------
        int
            expected words count in the field. Returns 0 if field is infinite.
        """
        return self.bytes_expected // self.word_length


# todo: __str__ method
# todo: typing - set content with bytes
# todo: try to copy docstrings (mypy)
class BytesField(BytesFieldABC[BytesFieldStruct]):
    """
    Represents parser for work with field content.

    Parameters
    ----------
    storage : ContinuousBytesStorage
        storage of fields.
    name : str
        field name.
    struct : BytesFieldStruct
        field struct instance
    """

    def __init__(
        self,
        storage: ContinuousBytesStorage,
        name: str,
        struct: BytesFieldStruct,
    ) -> None:
        super().__init__(name, struct)
        self._storage = storage

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self._storage.content[self._struct.slice_]


class ContinuousBytesStorage(
    BytesStorageABC[BytesField, BytesFieldStruct]
):
    """
    Represents continuous storage where data storage in bytes.

    Continuous means that the storage fields must go continuously, that is,
    where one field ends, another must begin.

    Parameters
    ----------
    name: str
        name of storage.
    fields: dict[str, BytesFieldStruct]
        fields of the storage. The kwarg Key is used as the field name.
    """

    def __init__(self, name: str, fields: dict[str, BytesFieldStruct]):
        for f_name, field in fields.items():
            if not isinstance(field, BytesFieldStruct):
                raise TypeError(f"invalid type of '{f_name}': {type(field)}")

        super().__init__(name, fields)

    def __getitem__(self, field: str) -> BytesField:
        """Get field parser."""
        return BytesField(self, field, self._f[field])


# todo: typehint - Generic. Create via generic for children.
# todo: to metaclass (because uses for generate new class)
# todo: wait to 3.12 for TypeVar with default type.
class BytesFieldPattern(PatternABC[BytesFieldStruct]):
    """
    Represents class which storage common parameters for field.
    """

    _options = {"base": BytesFieldStruct}

    def __init__(self, typename: str = "base", **parameters: Any) -> None:
        super().__init__(typename=typename, **parameters)

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> BytesFieldStruct:
        """
        Get field initialized with pattern and `additions`.


        Parameters
        ----------
        changes_allowed : bool, default=False
            if False intersection between `additions` and pattern is
            prohibited.
        additions : Any
            additional arguments for field.

        Returns
        -------
        BytesFieldStruct
            initialized field.

        Raises
        ------
        SyntaxError
            if `changes_allowed` is False and pattern and `additions` has
            intersection.
        """
        if not changes_allowed:
            intersection = set(self._kw).intersection(set(additions))
            if len(intersection) > 0:
                raise SyntaxError(
                    f"keyword argument(s) repeated: {', '.join(intersection)}"
                )

        kwargs = self._kw.copy()
        kwargs.update(additions)
        return self._target(**kwargs)


class BytesStoragePattern(
    ContinuousBytesStoragePatternABC[
        ContinuousBytesStorage, BytesFieldPattern, BytesFieldStruct
    ]
):
    """
    Represents pattern for bytes storage.
    """

    _options = {"continuous": ContinuousBytesStorage}

    _sub_p_type = BytesFieldPattern

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> ContinuousBytesStorage:
        """
        Get initialized class instance with parameters from pattern and from
        `additions`.

        Parameters
        ----------
        changes_allowed: bool, default = False
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection, otherwise the
            `additions` take precedence.
        **additions: Any
            additional initialization parameters. Those keys that are
            separated by "__" will be defined as parameters for other
            patterns target, otherwise for the storage target.

        Returns
        -------
        ContinuousBytesStorage
            initialized target class.

        Raises
        ------
        AssertionError
            if in some reason typename is invalid.
        NotConfiguredYet
            if patterns list is empty.
        """
        if len(self._sub_p) == 0:
            raise NotConfiguredYet(self)

        for_field, for_storage = split_complex_dict(
            additions, without_sep="other"
        )

        if not changes_allowed:
            intersection = set(self._kw).intersection(set(for_storage))
            if len(intersection) > 0:
                raise SyntaxError(
                    f"keyword argument(s) repeated: {', '.join(intersection)}"
                )

        if self._tn == "continuous":
            return self._get_continuous(
                changes_allowed, for_storage, for_field
            )
        raise AssertionError(f"invalid typename: '{self._tn}'")

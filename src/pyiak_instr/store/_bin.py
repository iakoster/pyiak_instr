"""Private module of ``pyiak_instr.store`` for work with bytes."""
from __future__ import annotations

from typing import (
    Any,
    Iterable,
)

import numpy as np
import numpy.typing as npt

from ..encoders import BytesEncoder
from ..types.store import (
    STRUCT_DATACLASS,
    BytesFieldABC,
    BytesFieldPatternABC,
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


@STRUCT_DATACLASS
class BytesFieldStruct(BytesFieldStructProtocol):
    """
    Represents field parameters with values encoded in bytes.
    """

    def __post_init__(self) -> None:
        BytesEncoder.check_fmt_order(self.fmt, self.order)
        super().__post_init__()

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

    @property
    def word_bytesize(self) -> int:
        """
        Returns
        -------
        int
            The length of the one word in bytes.
        """
        return BytesEncoder.get_bytesize(self.fmt)


# todo: __str__ method
# todo: typing - set content with bytes
# todo: try to copy docstrings (mypy)
class BytesField(BytesFieldABC["ContinuousBytesStorage", BytesFieldStruct]):
    """
    Represents parser for work with field content.
    """


class ContinuousBytesStorage(
    BytesStorageABC["BytesStoragePattern", BytesField, BytesFieldStruct]
):
    """
    Represents continuous storage where data storage in bytes.

    Continuous means that the storage fields must go continuously, that is,
    where one field ends, another must begin.
    """

    _struct_field = {BytesFieldStruct: BytesField}


# todo: typehint - Generic. Create via generic for children.
# todo: to metaclass (because uses for generate new class)
# todo: wait to 3.12 for TypeVar with default type.
class BytesFieldPattern(BytesFieldPatternABC[BytesFieldStruct]):
    """
    Represents class which storage common parameters for field.
    """

    _options = {"basic": BytesFieldStruct}


class BytesStoragePattern(
    ContinuousBytesStoragePatternABC[
        ContinuousBytesStorage, BytesFieldPattern
    ]
):
    """
    Represents pattern for bytes storage.
    """

    _options = {"continuous": ContinuousBytesStorage}

    _sub_p_type = BytesFieldPattern

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

        Raises
        ------
        AssertionError
            if in some reason `typename` not in options.
        """
        if self._tn == "continuous":
            return super()._modify_all(changes_allowed, for_subs)
        raise AssertionError(f"invalid typename: '{self._tn}'")

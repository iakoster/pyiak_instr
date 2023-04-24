"""Private module of ``pyiak_instr.store`` for work with bytes."""
from __future__ import annotations

from typing import (
    Any,
    ClassVar,
    Iterable,
    TypeAlias,
)

import numpy as np
import numpy.typing as npt

from ..core import Code
from ..utilities import BytesEncoder, split_complex_dict
from ..exceptions import NotConfiguredYet
from ..types import PatternABC
from ..types.store import (
    STRUCT_DATACLASS,
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


@STRUCT_DATACLASS
class BytesFieldStruct(BytesFieldStructProtocol):
    """
    Represents field parameters with values encoded in bytes.
    """

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

    def _verify_values_before_modifying(self) -> None:
        BytesEncoder.check_fmt_order(self.fmt, self.order)
        super()._verify_values_before_modifying()

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


class ContinuousBytesStorage(BytesStorageABC[BytesField, BytesFieldStruct]):
    """
    Represents continuous storage where data storage in bytes.

    Continuous means that the storage fields must go continuously, that is,
    where one field ends, another must begin.
    """

    _struct_field = {
        BytesFieldStruct: BytesField
    }


# todo: typehint - Generic. Create via generic for children.
# todo: to metaclass (because uses for generate new class)
# todo: wait to 3.12 for TypeVar with default type.
class BytesFieldPattern(PatternABC[BytesFieldStruct]):
    """
    Represents class which storage common parameters for field.
    """

    _options = {"basic": BytesFieldStruct}


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

    def _get(
            self,
            changes_allowed: bool,
            for_meta: dict[str, Any],
            for_sub: dict[str, dict[str, Any]],
    ) -> ContinuousBytesStorage:
        """
        Get initialized class instance with parameters from pattern and from
        `additions`.

        Parameters
        ----------
        changes_allowed: bool
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection.
        for_meta: dict[str, Any]:
            dictionary with parameters for meta pattern in format
            {PARAMETER: VALUE}.
        for_sub: dict[str, dict[str, Any]]
            dictionary with parameters for sub patterns in format
            {PATTERN: {PARAMETER: VALUE}}.

        Returns
        -------
        ContinuousBytesStorage
            initialized target class.

        Raises
        ------
        NotConfiguredYet
            if patterns list is empty.
        """
        if self._tn == "continuous":
            return self._get_continuous(changes_allowed, for_meta, for_sub)
        raise AssertionError(f"invalid typename: '{self._tn}'")

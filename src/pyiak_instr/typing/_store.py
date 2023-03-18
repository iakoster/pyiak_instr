"""Private module of ``pyiak_instr`` with types of store module."""
from abc import ABC
from typing import Generic, TypeVar


__all__ = ["BytesFieldParserABC"]


FieldT = TypeVar("FieldT")
StorageT = TypeVar("StorageT")


# pylint: disable=too-few-public-methods
class BytesFieldParserABC(ABC, Generic[StorageT, FieldT]):
    """
    Abstract base class parser for byte fields.

    Parameters
    ----------
    storage : StorageT
        storage of fields.
    name : str
        field name.
    field : FieldT
        field instance.
    """

    def __init__(self, storage: StorageT, name: str, field: FieldT) -> None:
        self._s = storage
        self._name = name
        self._f = field

    @property
    def fld(self) -> FieldT:
        """
        Returns
        -------
        BytesField
            field instance.
        """
        return self._f

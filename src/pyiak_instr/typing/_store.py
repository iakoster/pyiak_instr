"""Private module of ``pyiak_instr`` with types of store module."""
from abc import ABC
from typing import Generic, TypeVar


__all__ = ["BytesFieldABC"]


ParametersT = TypeVar("ParametersT")
StorageT = TypeVar("StorageT")


# pylint: disable=too-few-public-methods
class BytesFieldABC(ABC, Generic[StorageT, ParametersT]):
    """
    Abstract base class parser for byte fields.

    Parameters
    ----------
    storage : StorageT
        storage of fields.
    name : str
        field name.
    parameters : ParametersT
        parameters instance.
    """

    def __init__(
        self, storage: StorageT, name: str, parameters: ParametersT
    ) -> None:
        self._s = storage
        self._name = name
        self._p = parameters

    @property
    def parameters(self) -> ParametersT:
        """
        Returns
        -------
        FieldT
            parameters instance.
        """
        return self._p

"""Private module of ``pyiak_instr.types`` with encoder type."""
from abc import abstractmethod
from typing import Protocol, TypeVar, runtime_checkable


__all__ = ["Codec"]


DecodeT_co = TypeVar("DecodeT_co", covariant=True)
EncodeT_contra = TypeVar("EncodeT_contra", contravariant=True)
TargetT = TypeVar("TargetT")


@runtime_checkable
class Codec(Protocol[DecodeT_co, EncodeT_contra, TargetT]):
    """
    Represents abstract base class of encoder.
    """

    @abstractmethod
    def decode(self, data: TargetT) -> DecodeT_co:
        """
        Decode `data` to specified type.

        Parameters
        ----------
        data : TargetT
            data to decoding.

        Returns
        -------
        DecodeT_co
            decoded data.
        """

    @abstractmethod
    def encode(self, data: EncodeT_contra) -> TargetT:
        """
        Encode `data` to target type.

        Parameters
        ----------
        data : EncodeT_contra
            value to encoding.

        Returns
        -------
        TargetT
            encoded data.
        """

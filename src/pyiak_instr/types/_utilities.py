"""Private module of ``pyiak_instr.types`` with encoder type."""
from abc import abstractmethod
from typing import Protocol, TypeVar


__all__ = ["Encoder"]


DecodeT = TypeVar("DecodeT")
EncodeT = TypeVar("EncodeT")
TargetT = TypeVar("TargetT")


class Encoder(Protocol[DecodeT, EncodeT, TargetT]):
    """
    Represents abstract base class of encoder.
    """

    @abstractmethod
    def decode(self, value: TargetT) -> DecodeT:
        """
        Decode `value`.

        Parameters
        ----------
        value : TargetT
            value to decoding.

        Returns
        -------
        DecodeT
            decoded value.
        """

    @abstractmethod
    def encode(self, value: EncodeT) -> TargetT:
        """
        Encode `value`.

        Parameters
        ----------
        value : EncodeT
            value to encoding.

        Returns
        -------
        TargetT
            encoded value.
        """

"""Private module of ``pyiak_instr.types``."""
from abc import abstractmethod
from typing import Generic, TypeVar


__all__ = ["Decoder", "Encoder"]


DecodeT = TypeVar("DecodeT")
EncodeT = TypeVar("EncodeT")
TargetT = TypeVar("TargetT")


class Decoder(Generic[DecodeT, TargetT]):
    """
    Represents abstract base class of decoder.
    """

    @abstractmethod
    def decode(self, data: TargetT) -> DecodeT:
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


class Encoder(Generic[EncodeT, TargetT]):
    """
    Represents abstract base class of encoder.
    """

    @abstractmethod
    def encode(self, data: EncodeT) -> TargetT:
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

"""Private module of ``pyiak_instr.encoders.bin``."""
from struct import calcsize
from abc import ABC
from typing import Any, ClassVar, Iterable, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from ...core import Code
from ...exceptions import NotAmongTheOptions
from ..types import Decoder, Encoder


__all__ = [
    "BytesDecoder",
    "BytesEncoder",
    "BytesIntDecoder",
    "BytesIntEncoder",
    "BytesFloatDecoder",
    "BytesFloatEncoder",
    "get_bytes_transformers",
]


DecodeT = TypeVar("DecodeT")
EncodeT = TypeVar("EncodeT")


IntDecodeT = npt.NDArray[np.int_]
IntEncodeT = bytes | int | list[int] | npt.NDArray[np.int_]

_U = {
    Code.U8: 1,
    Code.U16: 2,
    Code.U24: 3,
    Code.U32: 4,
    Code.U40: 5,
    Code.U48: 6,
    Code.U56: 7,
    Code.U64: 8,
}

_I = {
    Code.I8: 1,
    Code.I16: 2,
    Code.I24: 3,
    Code.I32: 4,
    Code.I40: 5,
    Code.I48: 6,
    Code.I56: 7,
    Code.I64: 8,
}


FloatDecodeT = npt.NDArray[np.float_]
FloatEncodeT = bytes | float | list[float] | npt.NDArray[np.float_]

_F = {
    Code.F16: "e",
    Code.F32: "f",
    Code.F64: "d",
}


class BytesDecoder(ABC, Decoder[DecodeT, bytes]):
    """
    A base class for a byte decoder.
    """

    _vs: int

    @property
    def fmt_bytesize(self) -> int:
        """
        Returns
        -------
        int
            single value size in encoded view.
        """
        return self._vs


class BytesEncoder(ABC, Encoder[EncodeT, bytes]):
    """
    A base class for a byte encoder.
    """

    _vs: int

    @property
    def word_bytesize(self) -> int:
        """
        Returns
        -------
        int
            single value size in encoded view.
        """
        return self._vs


class BytesTransformerMixin:
    """
    Base mixin for bytes transformers (decoder/encoder).

    Parameters
    ----------
    fmt : Code
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder.
    """

    ALLOWED: ClassVar[set[Code]]

    def __init__(self, fmt: Code, order: Code = Code.BIG_ENDIAN) -> None:
        if fmt not in self.ALLOWED:
            raise NotAmongTheOptions("fmt", value=fmt)
        if order not in {Code.BIG_ENDIAN, Code.LITTLE_ENDIAN}:
            raise NotAmongTheOptions("order", value=order)

        self._fmt, self._order = fmt, order
        self._vs = 1


class BytesIntTransformerMixin(BytesTransformerMixin):
    """
    Mixin for int-bytes transformers (decoder/encoder).

    Parameters
    ----------
    fmt : Code, default=U8
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder.
    """

    ALLOWED = set(_U).union(set(_I))

    def __init__(
        self, fmt: Code = Code.U8, order: Code = Code.BIG_ENDIAN
    ) -> None:
        super().__init__(fmt, order=order)
        self._str_order: Literal["little", "big"] = (
            "big" if order is Code.BIG_ENDIAN else "little"
        )
        self._signed = fmt in _I
        self._vs = _I[fmt] if self._signed else _U[fmt]


class BytesIntDecoder(BytesDecoder[IntDecodeT], BytesIntTransformerMixin):
    """
    Decoder for integer or iterable of integers.
    """

    def decode(self, data: bytes) -> IntDecodeT:
        """
        Decode `data` from bytes.

        Parameters
        ----------
        data : bytes
            data for decoding.

        Returns
        -------
        IntDecodeT
            decoded data.
        """
        return np.fromiter(
            (
                int.from_bytes(
                    data[i : i + self._vs],
                    self._str_order,
                    signed=self._signed,
                )
                for i in range(0, len(data), self._vs)
            ),
            dtype=np.int_,
        )


class BytesIntEncoder(BytesEncoder[IntEncodeT], BytesIntTransformerMixin):
    """
    Encoder for integer or iterable of integers.
    """

    def encode(self, data: IntEncodeT) -> bytes:
        """
        Encode `data` to bytes.

        Parameters
        ----------
        data : IntEncodeT
            data to encoding.

        Returns
        -------
        bytes
            encoded data.
        """
        if isinstance(data, bytes):
            return data
        if not isinstance(data, Iterable | np.ndarray):
            data = [data]
        return b"".join(
            int(v).to_bytes(self._vs, self._str_order, signed=self._signed)
            for v in data
        )


class BytesFloatTransformerMixin(BytesTransformerMixin):
    """
    Mixin for float-bytes transformers (decoder/encoder).

    Parameters
    ----------
    fmt : Code, default=F32
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder.
    """

    ALLOWED = set(_F)

    def __init__(
        self, fmt: Code = Code.F32, order: Code = Code.BIG_ENDIAN
    ) -> None:
        super().__init__(fmt, order=order)
        self._vs = calcsize(_F[fmt])
        self._dtype = (">" if order is Code.BIG_ENDIAN else "<") + _F[fmt]


class BytesFloatDecoder(
    BytesDecoder[FloatDecodeT], BytesFloatTransformerMixin
):
    """
    Decoder for float or iterable of floats.
    """

    def decode(self, data: bytes) -> FloatDecodeT:
        """
        Decode `data` from bytes.

        Parameters
        ----------
        data : bytes
            data for decoding.

        Returns
        -------
        FloatDecodeT
            decoded data.
        """
        return np.frombuffer(data, dtype=self._dtype)


class BytesFloatEncoder(
    BytesEncoder[FloatEncodeT], BytesFloatTransformerMixin
):
    """
    Encoder for float or iterable of floats.
    """

    def encode(self, data: FloatEncodeT) -> bytes:
        """
        Encode `data` to bytes.

        Parameters
        ----------
        data : FloatEncodeT
            data to encoding.

        Returns
        -------
        bytes
            encoded data.
        """
        if isinstance(data, bytes):
            return data
        return np.array(data, dtype=self._dtype).tobytes()


def get_bytes_transformers(
    fmt: Code, order: Code = Code.BIG_ENDIAN
) -> tuple[BytesDecoder[Any], BytesEncoder[Any]]:
    """
    Get transformers to encoding to/decoding from bytes.

    Parameters
    ----------
    fmt : Code
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder.

    Returns
    -------
    tuple[BytesDecoder[Any], BytesEncoder[Any]]
        specified encoder and decoder.

    Raises
    ------
    ValueError
        if there is no transformer for specified `fmt`.
    """

    if fmt in BytesIntTransformerMixin.ALLOWED:
        return (
            BytesIntDecoder(fmt=fmt, order=order),
            BytesIntEncoder(fmt=fmt, order=order),
        )

    if fmt in BytesFloatTransformerMixin.ALLOWED:
        return (
            BytesFloatDecoder(fmt=fmt, order=order),
            BytesFloatEncoder(fmt=fmt, order=order),
        )

    raise ValueError(f"unsupported format: {fmt!r}")

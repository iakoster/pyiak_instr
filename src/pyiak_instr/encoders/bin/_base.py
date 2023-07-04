"""Private module of ``pyiak_instr.encoders.bin``."""
from struct import calcsize
from typing import Iterable, Literal

import numpy as np
import numpy.typing as npt

from ...core import Code
from ...exceptions import NotAmongTheOptions
from ..types import Encoder


__all__ = ["BytesEncoder", "BytesIntEncoder", "BytesFloatEncoder"]


IntDecodeT = npt.NDArray[np.int_]
IntEncodeT = int | list[int] | npt.NDArray[np.int_]


class BytesIntEncoder(Encoder[IntDecodeT, IntEncodeT, bytes]):
    """
    Encoder/Decoder for integer or iterable of integers.

    Parameters
    ----------
    fmt : Code, default=U8
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder.
    """

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

    ALLOWED = set(_U).union(set(_I))

    def __init__(
        self, fmt: Code = Code.U8, order: Code = Code.BIG_ENDIAN
    ) -> None:
        if fmt not in self.ALLOWED:
            raise NotAmongTheOptions("fmt", value=fmt)
        if order not in {Code.BIG_ENDIAN, Code.LITTLE_ENDIAN}:
            raise NotAmongTheOptions("order", value=order)

        self._order: Literal["little", "big"] = (
            "big" if order is Code.BIG_ENDIAN else "little"
        )
        self._signed = fmt in self._I
        self._size = self._I[fmt] if self._signed else self._U[fmt]

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
        bytesize = self.value_size
        encoded = np.empty(len(data) // bytesize, np.int_)
        for i in range(encoded.shape[0]):
            val = data[i * bytesize : (i + 1) * bytesize]
            encoded[i] = int.from_bytes(val, self._order, signed=self._signed)
        return encoded

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
        if isinstance(data, Iterable):
            size = self.value_size
            return b"".join(
                map(
                    lambda v: int(v).to_bytes(
                        size, self._order, signed=self._signed
                    ),
                    data,
                )
            )
        return int(data).to_bytes(
            self.value_size, self._order, signed=self._signed
        )

    @property
    def value_size(self) -> int:
        """
        Returns
        -------
        int
            single value bytesize.
        """
        return self._size


FloatDecodeT = npt.NDArray[np.float_]
FloatEncodeT = float | list[float] | npt.NDArray[np.float_]


class BytesFloatEncoder(Encoder[FloatDecodeT, FloatEncodeT, bytes]):
    """
    Encoder/Decoder for float or iterable of floats.

    Parameters
    ----------
    fmt : Code, default=F32
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder.
    """

    _F = {
        Code.F16: "e",
        Code.F32: "f",
        Code.F64: "d",
    }

    ALLOWED = set(_F)

    def __init__(
        self, fmt: Code = Code.F32, order: Code = Code.BIG_ENDIAN
    ) -> None:
        if fmt not in self.ALLOWED:
            raise NotAmongTheOptions("fmt", value=fmt)
        if order not in {Code.BIG_ENDIAN, Code.LITTLE_ENDIAN}:
            raise NotAmongTheOptions("order", value=order)

        self._size = calcsize(self._F[fmt])
        self._dtype = (">" if order is Code.BIG_ENDIAN else "<") + self._F[
            fmt
        ]

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
        return np.array(data, dtype=self._dtype).tobytes()

    @property
    def value_size(self) -> int:
        """
        Returns
        -------
        int
            single value bytesize.
        """
        return self._size


class BytesEncoder(
    Encoder[IntDecodeT | FloatDecodeT, IntEncodeT | FloatEncodeT, bytes]
):
    """
    Encoder/Decoder to/from bytes.

    Parameters
    ----------
    fmt : Code, default=U8
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder.
    """

    def __init__(
        self, fmt: Code = Code.U8, order: Code = Code.BIG_ENDIAN
    ) -> None:
        self._encoder: BytesIntEncoder | BytesFloatEncoder
        if fmt in BytesIntEncoder.ALLOWED:
            self._encoder = BytesIntEncoder(fmt, order)

        elif fmt in BytesFloatEncoder.ALLOWED:
            self._encoder = BytesFloatEncoder(fmt, order)

        else:
            raise ValueError(f"invalid fmt: {fmt!r}")

    def decode(self, data: bytes) -> IntDecodeT | FloatDecodeT:
        """
        Decode `data` from bytes.

        Parameters
        ----------
        data : bytes
            data for decoding.

        Returns
        -------
        IntDecodeT | FloatDecodeT
            decoded data.
        """
        return self._encoder.decode(data)

    def encode(self, data: IntEncodeT | FloatEncodeT) -> bytes:
        """
        Encode `data` to bytes.

        Parameters
        ----------
        data : IntEncodeT | FloatEncodeT
            data to encoding.

        Returns
        -------
        bytes
            encoded data.
        """
        return self._encoder.encode(data)  # type: ignore[arg-type]

    @property
    def value_size(self) -> int:
        """
        Returns
        -------
        int
            single value bytesize.
        """
        return self._encoder.value_size

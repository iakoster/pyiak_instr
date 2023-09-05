"""Private module of ``pyiak_instr.codecs.bin``."""
from __future__ import annotations
from struct import calcsize
from functools import wraps
from abc import ABC
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    TypeVar,
)

import numpy as np
import numpy.typing as npt

from ...core import Code
from ...exceptions import NotAmongTheOptions
from ..types import Codec


__all__ = [
    "BytesCodec",
    "BytesIntCodec",
    "BytesFloatCodec",
    "BytesHexCodec",
    "BytesStringCodec",
    "bytes_instant_return",
    "get_bytes_codec",
]


def bytes_instant_return(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Changing the method functionality so that it always returns bytes type
    when trying to encode them.

    Parameters
    ----------
    func: Callable[..., Any]
        method.

    Returns
    -------
    Callable[..., Any]
        wrapper
    """

    # pylint: disable=missing-return-doc,missing-return-type-doc
    @wraps(func)
    def wrapper(  # type: ignore[no-untyped-def]
        self: BytesCodec[Any, Any], data: Any, *args: Any, **kwargs: Any
    ):
        if isinstance(data, bytes):
            return data
        return func(self, data, *args, **kwargs)

    return wrapper


DecodeT_co = TypeVar("DecodeT_co", covariant=True)
EncodeT_contra = TypeVar("EncodeT_contra", contravariant=True)


class BytesCodec(
    ABC,
    Codec[DecodeT_co, EncodeT_contra, bytes],
    Generic[DecodeT_co, EncodeT_contra],
):
    """
    Base class of codec to encoding/decoding bytes.

    Parameters
    ----------
    fmt : Code
        format of single value.
    """

    ALLOWED: ClassVar[set[Code]]

    def __init__(self, fmt: Code) -> None:
        if fmt not in self.ALLOWED:
            raise NotAmongTheOptions("fmt", value=fmt)
        self._fmt = fmt
        self._fmt_size = 1

    @property
    def fmt_bytesize(self) -> int:
        """
        Returns
        -------
        int
            single value bytesize.
        """
        return self._fmt_size


IntDecodeT = npt.NDArray[np.int_]
IntEncodeT = bytes | int | list[int] | npt.NDArray[np.int_]


class BytesIntCodec(BytesCodec[IntDecodeT, IntEncodeT]):
    """
    Codec for integer or iterable of integers.

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
        super().__init__(fmt)

        if order not in {Code.BIG_ENDIAN, Code.LITTLE_ENDIAN}:
            raise NotAmongTheOptions("order", value=order)
        self._order = order

        self._str_order: Literal["little", "big"] = (
            "big" if order is Code.BIG_ENDIAN else "little"
        )
        self._signed = fmt in self._I
        self._fmt_size = self._I[fmt] if self._signed else self._U[fmt]

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
                    data[i : i + self._fmt_size],
                    self._str_order,
                    signed=self._signed,
                )
                for i in range(0, len(data), self._fmt_size)
            ),
            dtype=np.int_,
        )

    @bytes_instant_return
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
        if not isinstance(data, Iterable | np.ndarray):
            data = [data]
        return b"".join(
            int(v).to_bytes(
                self._fmt_size, self._str_order, signed=self._signed
            )
            for v in data
        )


FloatDecodeT = npt.NDArray[np.float_]
FloatEncodeT = bytes | float | list[float] | npt.NDArray[np.float_]


class BytesFloatCodec(BytesCodec[FloatDecodeT, FloatEncodeT]):
    """
    Codec for float or iterable of floats.

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
        super().__init__(fmt)

        if order not in {Code.BIG_ENDIAN, Code.LITTLE_ENDIAN}:
            raise NotAmongTheOptions("order", value=order)
        self._order = order

        str_fmt = self._F[fmt]
        self._fmt_size = calcsize(str_fmt)
        self._dtype = (">" if order is Code.BIG_ENDIAN else "<") + str_fmt

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

    @bytes_instant_return
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


HexDecodeT = str
HexEncodeT = bytes | str


class BytesHexCodec(BytesCodec[HexDecodeT, HexEncodeT]):
    """
    Codec for hex string.

    It is the responsibility of the user to ensure that the input data is
    correct (e.g., insignificant zeros).

    Parameters
    ----------
    fmt : Code, default=HEX
        format of single value.
    """

    ALLOWED = {Code.HEX}

    def __init__(self, fmt: Code = Code.HEX) -> None:
        super().__init__(fmt=fmt)

    def decode(self, data: bytes) -> HexDecodeT:
        """
        Decode `data` from bytes.

        Parameters
        ----------
        data : bytes
            data for decoding.

        Returns
        -------
        HexDecodeT
            decoded data.
        """
        return data.hex()

    @bytes_instant_return
    def encode(self, data: HexEncodeT) -> bytes:
        """
        Encode `data` to bytes.

        Parameters
        ----------
        data : HexEncodeT
            data to encoding.

        Returns
        -------
        bytes
            encoded data.
        """
        assert not isinstance(data, bytes), "bytes not allowed"
        return bytes.fromhex(data)


StringDecodeT = str
StringEncodeT = bytes | str


class BytesStringCodec(BytesCodec[StringDecodeT, StringEncodeT]):
    """
    Codec for string.

    Parameters
    ----------
    fmt : Code, default=STRING
        format of single value.
    encoding : str, default='ascii'
        string encoding.
    """

    ALLOWED = {Code.STRING}

    def __init__(
        self, fmt: Code = Code.STRING, encoding: str = "ascii"
    ) -> None:
        super().__init__(fmt)
        self._encoding = encoding

    def decode(self, data: bytes) -> StringDecodeT:
        """
        Decode `data` from bytes.

        Parameters
        ----------
        data : bytes
            data for decoding.

        Returns
        -------
        StringDecodeT
            decoded data.
        """
        return data.decode(encoding=self._encoding)

    @bytes_instant_return
    def encode(self, data: StringEncodeT) -> bytes:
        """
        Encode `data` to bytes.

        Parameters
        ----------
        data : StringEncodeT
            data to encoding.

        Returns
        -------
        bytes
            encoded data.
        """
        assert not isinstance(data, bytes), "bytes not allowed"
        return data.encode(encoding=self._encoding)


def get_bytes_codec(
    fmt: Code,
    order: Code = Code.BIG_ENDIAN,
    encoding: str = "ascii",
) -> BytesCodec[Any, Any]:
    """
    Get codec to encoding to/decoding from bytes.

    Parameters
    ----------
    fmt : Code
        format of single value.
    order : Code, default=BIG_ENDIAN
        byteorder. Used for:

            * BytesIntCodec;
            * BytesFloatCodec.
    encoding : str, default='ascii'
        encoding for strings. Used for:

            * BytesStringCodec

    Returns
    -------
    BytesCodec[Any, Any]
        specified codec.

    Raises
    ------
    ValueError
        if there is no codec for specified format.
    """
    if fmt in BytesIntCodec.ALLOWED:
        return BytesIntCodec(fmt=fmt, order=order)

    if fmt in BytesFloatCodec.ALLOWED:
        return BytesFloatCodec(fmt=fmt, order=order)

    if fmt in BytesHexCodec.ALLOWED:
        return BytesHexCodec(fmt=fmt)

    if fmt in BytesStringCodec.ALLOWED:
        return BytesStringCodec(fmt=fmt, encoding=encoding)

    raise ValueError(f"unsupported format: {fmt!r}")

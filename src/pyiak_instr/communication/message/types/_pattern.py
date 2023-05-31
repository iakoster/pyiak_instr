"""Private module of ``pyiak_instr.communication.message``."""
from abc import abstractmethod
from typing import (
    Any,
    Self,
    TypeVar,
)

from ....core import Code
from ....store.bin.types import (
    BytesFieldStructPatternABC,
    BytesStorageStructPatternABC,
    BytesStoragePatternABC,
)
from ._struct import MessageFieldStructABC, MessageStructABC
from ._message import MessageABC


__all__ = [
    "MessageFieldStructPatternABC",
    "MessageStructPatternABC",
    "MessagePatternABC",
]


FieldStructT = TypeVar("FieldStructT", bound=MessageFieldStructABC)
MessageStructT = TypeVar("MessageStructT", bound=MessageStructABC[Any])
MessageT = TypeVar("MessageT", bound=MessageABC[Any, Any, Any, Any])

FieldStructPatternT = TypeVar(
    "FieldStructPatternT", bound="MessageFieldStructPatternABC[Any]"
)
MessageStructPatternT = TypeVar(
    "MessageStructPatternT", bound="MessageStructPatternABC[Any, Any]"
)


class MessageFieldStructPatternABC(BytesFieldStructPatternABC[FieldStructT]):
    """
    Represent base class of pattern for field struct.
    """

    @staticmethod
    @abstractmethod
    def get_fmt_bytesize(fmt: Code) -> int:
        """
        Get fmt size in bytes.

        Parameters
        ----------
        fmt : Code
            fmt code.

        Returns
        -------
        int
            fmt bytesize.
        """

    @classmethod
    def basic(
        cls,
        typename: str = "basic",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        bytes_expected: int = 0,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for basic field.

        Parameters
        ----------
        typename : str, default='basic'
            basic typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        bytes_expected : int, default=0
            expected count of bytes.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename=typename,
            fmt=fmt,
            order=order,
            bytes_expected=bytes_expected,
            default=default,
        )

    @classmethod
    def single(
        cls,
        typename: str = "single",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for single field.

        Parameters
        ----------
        typename : str, default='single'
            single typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

    @classmethod
    def static(
        cls,
        typename: str = "static",
        default: bytes = b"\x00",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
    ) -> Self:
        """
        Get initialized pattern for static field.

        Parameters
        ----------
        typename : str, default='static'
            static typename.
        fmt : Code, default=Code.U8
            value format.
        default : bytes, default=b'\x00'
            default value for field.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

    @classmethod
    def address(
        cls,
        typename: str = "address",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        behaviour: Code = Code.DMA,
        units: Code = Code.WORDS,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for address field.

        Parameters
        ----------
        typename : str, default='address'
            address typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        behaviour : Code, default=Code.DMA
            address field behaviour.
        units : Code, default=Code.WORDS
            address units.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            behaviour=behaviour,
            units=units,
            default=default,
        )

    @classmethod
    def crc(
        cls,
        typename: str = "crc",
        fmt: Code = Code.U16,
        order: Code = Code.BIG_ENDIAN,
        poly: int = 0x1021,
        init: int = 0,
        default: bytes = b"",
        fill_value: bytes = b"\x00",
        wo_fields: set[str] | None = None,
    ) -> Self:
        """
        Get initialized pattern for crc field.

        Parameters
        ----------
        typename : str, default='crc'
            crc typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        poly : int, default=0x1021
            poly for crc algorithm.
        init : int, default=0
            init value for crc algorithm.
        default : bytes, default=b''
            default value for field.
        fill_value : bytes, default=b''
            fill value for field
        wo_fields : set[str] | None, default=None
            a set of field names that are not used to calculate the crc.

        Returns
        -------
        Self
            initialized pattern.
        """
        if wo_fields is None:
            wo_fields = set()
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            poly=poly,
            init=init,
            default=default,
            fill_value=fill_value,
            wo_fields=wo_fields,
        )

    @classmethod
    def data(
        cls,
        typename: str = "data",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        bytes_expected: int = 0,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for data field.

        Parameters
        ----------
        typename : str, default='data'
            data typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        bytes_expected : int, default=0
            expected count of bytes.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename=typename,
            fmt=fmt,
            order=order,
            bytes_expected=bytes_expected,
            default=default,
        )

    @classmethod
    def data_length(
        cls,
        typename: str = "data_length",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        behaviour: Code = Code.ACTUAL,
        units: Code = Code.BYTES,
        additive: int = 0,
        default: bytes = b"",
        fill_value: bytes = b"\x00",
    ) -> Self:
        """
        Get initialized pattern for data length field.

        Parameters
        ----------
        typename : str, default='data_length'
            data length typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        behaviour: Code, default=Code.ACTUAL
            data length field behaviour.
        units: Code, default=Code.BYTES
            data length units.
        additive: int, default=0
            additive value for data length value.
        default : bytes, default=b''
            default value for field.
        fill_value : bytes, default=b''
            fill value for field

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            behaviour=behaviour,
            units=units,
            additive=additive,
            default=default,
            fill_value=fill_value,
        )

    @classmethod
    def id_(
        cls,
        typename: str = "id",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for id field.

        Parameters
        ----------
        typename : str, default='id'
            id typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

    @classmethod
    def operation(
        cls,
        typename: str = "operation",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        descs: dict[int, Code] | None = None,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for crc field.

        Parameters
        ----------
        typename : str, default='operation'
            operation typename.
        fmt : Code, default=Code.U8
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        descs: dict[int, Code] | None, default=None
            operation value descriptions.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        if descs is None:
            descs = {0: Code.READ, 1: Code.WRITE}
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            descs=descs,
            default=default,
        )

    @classmethod
    def response(
        cls,
        typename: str = "response",
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        descs: dict[int, Code] | None = None,
        default: bytes = b"",
    ) -> Self:
        """
        Get initialized pattern for crc field.

        Parameters
        ----------
        typename : str, default='response'
            response typename.
        fmt : Code
            value format.
        order : Code, default=Code.BIG_ENDIAN
            value byte order.
        descs: dict[Code, int] | None, default=None
            response value descriptions.
        default : bytes, default=b''
            default value for field.

        Returns
        -------
        Self
            initialized pattern.
        """
        if descs is None:
            descs = {}
        return cls(
            typename=typename,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            descs=descs,
            default=default,
        )


class MessageStructPatternABC(
    BytesStorageStructPatternABC[MessageStructT, FieldStructPatternT]
):
    """
    Represent base class of pattern for message struct.
    """

    @classmethod
    def basic(
        cls,
        typename: str = "basic",
        divisible: bool = False,
        mtu: int = 1500,
    ) -> Self:
        """
        Get initialized pattern for basic storage struct.

        Parameters
        ----------
        typename : str, default='basic'
            basic typename.
        divisible : bool, default=False
            shows that the message can be divided by the infinite field.
        mtu : int, default=1500
            max size of one message part.

        Returns
        -------
        Self
            initialized self instance.
        """
        return cls(typename=typename, divisible=divisible, mtu=mtu)


class MessagePatternABC(
    BytesStoragePatternABC[MessageT, MessageStructPatternT]
):
    """
    Represent base class of pattern for message.
    """

    @classmethod
    def basic(cls, typename: str = "basic") -> Self:
        """
        Get initialized pattern for basic message.

        Parameters
        ----------
        typename : str, default='basic'
            basic typename.

        Returns
        -------
        Self
            initialized self instance.
        """
        return cls(typename=typename)

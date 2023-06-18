"""Private module of ``pyiak_instr.communication.message``."""
from abc import abstractmethod
from typing import (
    Any,
    Self,
    TypeVar,
)

from ...core import Code
from ...encoders import BytesEncoder
from ...exceptions import NotAmongTheOptions, NotConfiguredYet
from ...types import Additions
from ...store.bin import (
    FieldPattern as BinFieldPattern,
    ContinuousStructPattern as BinStructPattern,
    ContainerPattern as BinContainerPattern,
)
from ._struct import Basic, Struct
from ._message import Message


__all__ = [
    "FieldPattern",
    "StructPattern",
    "MessagePattern",
]


FieldT = TypeVar("FieldT", bound=Basic)
StructT = TypeVar("StructT", bound=Struct[Any])
MessageT = TypeVar("MessageT", bound=Message[Any, Any, Any, Any])

FieldPatternT = TypeVar("FieldPatternT", bound="FieldPattern[Any]")
StructPatternT = TypeVar("StructPatternT", bound="StructPattern[Any, Any]")


class FieldPattern(BinFieldPattern[FieldT]):
    """
    Represent base class of pattern for field struct.

    Parameters
    ----------
    typename : str
        name of pattern target type.
    direction : Code, default=Code.ANY
        the direction of the field that indicates what type of message this
        message is for.
    **parameters : Any
        parameters for target initialization.
    """

    def __init__(
        self, typename: str, direction: Code = Code.ANY, **parameters: Any
    ) -> None:
        if direction not in {Code.ANY, Code.RX, Code.TX}:
            raise NotAmongTheOptions(
                "direction", direction, {Code.ANY, Code.RX, Code.TX}
            )

        self._dir = direction
        super().__init__(typename, **parameters)

    @classmethod
    def basic(
        cls,
        typename: str = "basic",
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
            fmt=fmt,
            order=order,
            bytes_expected=bytes_expected,
            default=default,
        )

    @classmethod
    def static(
        cls,
        typename: str = "static",
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

    @classmethod
    def address(
        cls,
        typename: str = "address",
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
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
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
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
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
            fmt=fmt,
            order=order,
            bytes_expected=bytes_expected,
            default=default,
        )

    @classmethod
    def dynamic_length(
        cls,
        typename: str = "dynamic_length",
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
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
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            default=default,
        )

    @classmethod
    def operation(
        cls,
        typename: str = "operation",
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
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
        direction: Code = Code.ANY,
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
        direction : Code, default=Code.ANY
            field direction.
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
            direction=direction,
            bytes_expected=cls.get_fmt_bytesize(fmt),
            fmt=fmt,
            order=order,
            descs=descs,
            default=default,
        )

    @staticmethod
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
        return BytesEncoder(fmt=fmt).value_size

    @property
    def direction(self) -> Code:
        """
        Returns
        -------
        Code
            field direction.
        """
        return self._dir

    def __init_kwargs__(self) -> dict[str, Any]:
        init_kw = dict(
            typename="",
            direction=self._dir,
        )
        init_kw.update(super().__init_kwargs__())
        return init_kw


class StructPattern(BinStructPattern[StructT, FieldPatternT]):
    """
    Represent base class of pattern for message struct.
    """

    def instance_for_direction(self, direction: Code) -> Self:
        """
        Get `self` instance with fields specified for one specific direction.

        Parameters
        ----------
        direction : Code
            fields direction.

        Returns
        -------
        Self
            `self` instance with fields for `direction`.

        Raises
        ------
        NotConfiguredYet
            if pattern without fields.
        ValueError
            if direction not in {ANY, RX, TX}.
        """
        if len(self._sub_p) == 0:
            raise NotConfiguredYet(self)

        if direction is Code.ANY:
            return self

        if direction is Code.RX:
            not_allowed = Code.TX
        elif direction is Code.TX:
            not_allowed = Code.RX
        else:
            raise ValueError(f"invalid direction: {direction!r}")

        return self.__class__(**self.__init_kwargs__()).configure(
            **{
                n: p
                for n, p in self._sub_p.items()
                if p.direction is not not_allowed
            }
        )

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


class MessagePattern(BinContainerPattern[MessageT, StructPatternT]):
    """
    Represent base class of pattern for message.
    """

    def get_for_direction(
        self,
        direction: Code,
        additions: Additions = Additions(),
    ) -> MessageT:
        """
        Get message instance with fields for specified direction.

        Parameters
        ----------
        direction : Code
            direction for fields.
        additions: Additions, default=Additions()
            additional initialization parameters.

        Returns
        -------
        MessageT
            message witch specified for direction.
        """
        (name,) = self._sub_p.keys()
        storage = self._sub_p[name].instance_for_direction(direction)
        pattern = self.__class__(**self.__init_kwargs__()).configure(
            **{name: storage}
        )

        instance = pattern.get(additions=additions)
        object.__setattr__(instance, "_p", self)  # todo: refactor
        return instance

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

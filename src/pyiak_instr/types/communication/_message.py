"""Private module of ``pyiak_instr.types.communication`` with types for
communication module."""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar  # pylint: disable=unused-import

from ...core import Code
from ..store import (
    BytesFieldABC,
    BytesFieldPatternABC,
    BytesFieldStructProtocol,
    BytesStorageABC,
    ContinuousBytesStoragePatternABC,
)


__all__ = [
    "MessageABC",
    "MessageFieldABC",
    "MessageFieldPatternABC",
    "MessageGetParserABC",
    "MessageHasParserABC",
    "MessagePatternABC",
]


AddressT = TypeVar("AddressT")
StructT = TypeVar("StructT", bound=BytesFieldStructProtocol)
FieldT = TypeVar("FieldT", bound="MessageFieldABC[Any, Any]")
FieldAnotherT = TypeVar("FieldAnotherT", bound="MessageFieldABC[Any, Any]")
MessageGetParserT = TypeVar(
    "MessageGetParserT", bound="MessageGetParserABC[Any, Any]"
)
MessageHasParserT = TypeVar(
    "MessageHasParserT", bound="MessageHasParserABC[Any]"
)
MessageT = TypeVar("MessageT", bound="MessageABC[Any, Any, Any, Any]")
FieldPatternT = TypeVar("FieldPatternT", bound="MessageFieldPatternABC[Any]")
MessagePatternT = TypeVar(
    "MessagePatternT", bound="MessagePatternABC[Any, Any]"
)


class MessageFieldABC(
    BytesFieldABC[MessageT, StructT], Generic[MessageT, StructT]
):
    """
    Represents abstract class for message field parser.
    """


class MessageGetParserABC(ABC, Generic[MessageT, FieldT]):
    """
    Abstract base class for parser to get the field from message by it type.

    Parameters
    ----------
    message: MessageT
        message instance.
    types: dict[type[FieldT], str]
        dictionary of field types in the message.
    """

    def __init__(
        self, message: MessageT, types: dict[type[FieldT], str]
    ) -> None:
        self._msg, self._types = message, types

    def __call__(self, type_: type[FieldAnotherT]) -> FieldAnotherT:
        """Get first field with specified type."""
        if type_ not in self._types:
            raise TypeError(f"{type_.__name__} instance is not found")
        return self._msg[  # type: ignore[no-any-return]
            self._types[type_]  # type: ignore[index]
        ]


class MessageHasParserABC(ABC, Generic[FieldT]):
    """
    Abstract base class parser to check the field class exists in the message.

    Parameters
    ----------
    types: set[type[FieldT]]
        set of field types in the message.
    """

    def __init__(self, types: set[type[FieldT]]) -> None:
        self._types = types

    def __call__(self, type_: type[FieldT]) -> bool:
        """Check that message has field of specified type."""
        return type_ in self._types


class MessageABC(
    BytesStorageABC[FieldT, StructT],
    Generic[
        FieldT,
        StructT,
        MessageGetParserT,
        MessageHasParserT,
        AddressT,
    ],
):
    """
    Abstract base class message for communication between devices.

    Parameters
    ----------
    name: str, default='std'
        name of storage.
    divisible: bool, default=False
        shows that the message can be divided by the infinite field.
    mtu: int, default=1500
        max size of one message part.
    **fields: StructT
        fields of the storage. The kwarg Key is used as the field name.
    """

    _get_parser: type[MessageGetParserT]
    _has_parser: type[MessageHasParserT]

    def __init__(
        self,
        name: str = "std",
        divisible: bool = False,
        mtu: int = 1500,
        **fields: StructT,
    ) -> None:
        super().__init__(name, fields)
        self._div = divisible
        self._mtu = mtu

        self._types: dict[type[FieldT], str] = {}
        for f_name, struct in self._f.items():
            f_class = self._struct_field[struct.__class__]
            if f_class not in self._types:
                self._types[f_class] = f_name

        self._src, self._dst = None, None

        if divisible and mtu < self.minimum_size:
            raise ValueError("MTU cannot be less than the minimum size")

    @property
    def divisible(self) -> bool:
        """
        Returns
        -------
        bool
            shows that the message can be divided by the infinite field.
        """
        return self._div

    @property
    def dst(self) -> AddressT | None:
        """
        Returns
        -------
        AddressT | None
            destination address.
        """
        return self._dst

    @dst.setter
    def dst(self, destination: AddressT | None) -> None:
        """
        Set destination address.

        Parameters
        ----------
        destination : AddressT | None
            destination address.
        """
        self._dst = destination

    @property
    def get(self) -> MessageGetParserT:
        """
        Returns
        -------
        MessageGet
            get parser instance.
        """
        return self._get_parser(self, self._types)

    @property
    def has(self) -> MessageHasParserT:
        """
        Returns
        -------
        MessageHas
            get parser instance.
        """
        return self._has_parser(set(self._types))

    @property
    def mtu(self) -> int:
        """
        Returns
        -------
        int
            max size of one message part.
        """
        return self._mtu

    @property
    def src(self) -> AddressT | None:
        """
        Returns
        -------
        AddressT | None
            source address.
        """
        return self._src

    @src.setter
    def src(self, source: AddressT | None) -> None:
        """
        Set source address.

        Parameters
        ----------
        source : AddressT | None
            source address.
        """
        self._src = source

    @property
    def src_dst(self) -> tuple[AddressT | None, AddressT | None]:
        """
        Returns
        -------
        tuple[AddressT | None, AddressT | None]
            src - source address;
            dst - destination address.
        """
        return self.src, self.dst

    @src_dst.setter
    def src_dst(
            self, src_dst: tuple[AddressT | None, AddressT | None]
    ) -> None:
        """
        Set source and destination addresses.

        Parameters
        ----------
        src_dst : tuple[AddressT | None, AddressT | None]
            src - source address;
            dst - destination address.
        """
        self.src, self.dst = src_dst


class MessageFieldPatternABC(BytesFieldPatternABC[StructT], Generic[StructT]):
    """
    Represent abstract class of pattern for message field.
    """

    @staticmethod
    @abstractmethod
    def get_bytesize(fmt: Code) -> int:
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


class MessagePatternABC(
    ContinuousBytesStoragePatternABC[MessageT, FieldPatternT],
    Generic[MessageT, FieldPatternT],
):
    """
    Represent abstract class of pattern for message.
    """

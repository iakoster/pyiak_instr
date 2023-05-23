"""Private module of ``pyiak_instr.types.communication`` with types for
communication module."""
from abc import ABC, abstractmethod
from dataclasses import field as _field
from functools import wraps
from typing import (  # pylint: disable=unused-import
    Any,
    Callable,
    Generator,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from ....core import Code
from ....types import Encoder
from ....types.store.bin import (
    BytesDecodeT,
    BytesEncodeT,
    BytesStorageABC,
)
from ._struct import (
    STRUCT_DATACLASS,
    MessageFieldStructABC,
    MessageStructGetParserABC,
    MessageStructHasParserABC,
    MessageStructABC,
)


__all__ = [
    "MessageABC",
]


AddressT = TypeVar("AddressT")
FieldStructT = TypeVar("FieldStructT", bound=MessageFieldStructABC)
MessageStructT = TypeVar(
    "MessageStructT", bound=MessageStructABC[MessageFieldStructABC]
)
MessagePatternT = TypeVar("MessagePatternT")

# StructT = TypeVar("StructT") # , bound=BytesFieldStructProtocol)
# FieldT = TypeVar("FieldT", bound="MessageFieldABC[Any, Any]")
# FieldAnotherT = TypeVar("FieldAnotherT", bound="MessageFieldABC[Any, Any]")
# MessageGetParserT = TypeVar(
#     "MessageGetParserT", bound="MessageGetParserABC[Any, Any]"
# )
# MessageHasParserT = TypeVar(
#     "MessageHasParserT", bound="MessageHasParserABC[Any]"
# )
# MessageT = TypeVar(
#     "MessageT", bound="MessageABC[Any, Any, Any, Any, Any, Any]"
# )
# FieldPatternT = TypeVar("FieldPatternT", bound="MessageFieldPatternABC[Any]")
# MessagePatternT = TypeVar(
#     "MessagePatternT", bound="MessagePatternABC[Any, Any]"
# )


# todo: clear src and dst?
# todo: get rx and tx class instance
# todo: field parser
class MessageABC(
    BytesStorageABC[FieldStructT, MessageStructT, MessagePatternT],
    Generic[FieldStructT, MessageStructT, MessagePatternT, AddressT],
):

    def __init__(
            self,
            storage: MessageStructT,
            pattern: MessagePatternT | None = None,
    ):
        super().__init__(storage, pattern=pattern)
        if self._s.divisible and self.has.address:
            behaviour = self.get.address.behaviour

            if behaviour is not Code.DMA:
                raise TypeError(
                    "invalid address behaviour for divisible message: "
                    f"{behaviour!r}"
                )

        self._src, self._dst = None, None

    def autoupdate_fields(self) -> None:
        ...

    def split(self) -> Generator[Self, None, None]:
        """
        Split the message into parts over an infinite field.

        Yields
        ------
        Self
            message part.
        """
        if len(self) == 0:
            raise ValueError("message is empty")

        if not self._s.divisible or len(self) <= self._s.mtu:
            yield self
            return

        dyn_name = self._s.dynamic_field_name
        dyn = self._s[dyn_name]
        dyn_step = self._s.mtu - self._s.minimum_size
        # todo: useless if address units is Code.BYTES
        dyn_step -= dyn_step % dyn.word_bytesize

        has_address = self.has.address
        address_name = self.get.address.name if has_address else ""
        address = self.decode(address_name)[0] if has_address else -1

        # todo: useless if units of address is Code.BYTES
        if dyn_step % dyn.word_bytesize != 0:
            raise ValueError(
                "step of dynamic field is not even: "
                f"{dyn_step}/{dyn.word_bytesize}"
            )
        address_word_step = dyn_step // dyn.word_bytesize

        for i, dyn_start in enumerate(
                range(0, self.bytes_count(dyn_name), dyn_step)
        ):
            part = self.__class__(self._s, self._p)  # todo: vulnerability?

            content = b""
            for field in self._s:
                name = field.name

                if name == dyn_name:
                    content += self.content(
                        name
                    )[dyn_start : dyn_start + dyn_step]

                elif name == address_name:
                    address_field = self.get.address
                    address_units = address_field.units

                    if address_units is Code.BYTES:
                        address += dyn_step

                    elif address_units is Code.WORDS:
                        address += address_word_step

                    else:
                        raise ValueError(
                            f"undefined address behaviour: {address_units}"
                        )

                    content += address_field.encode(address, verify=True)

                else:
                    content += self.content(name)

            part.encode(content)
            part.autoupdate_fields()
            yield part

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
    def get(self) -> MessageStructGetParserABC[MessageStructT, FieldStructT]:
        """
        Returns
        -------
        MessageStructGetParserABC
            message struct get parser.
        """
        return self._s.get

    @property
    def has(self) -> MessageStructHasParserABC[MessageStructT, FieldStructT]:
        """
        Returns
        -------
        MessageStructHasParserABC
            message struct has parser.
        """
        return self._s.has

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


# class MessageFieldPatternABC(Generic[StructT]):  # BytesFieldPatternABC[StructT]
#     """
#     Represent abstract class of pattern for message field.
#     """
#
#     @staticmethod
#     @abstractmethod
#     def get_bytesize(fmt: Code) -> int:
#         """
#         Get fmt size in bytes.
#
#         Parameters
#         ----------
#         fmt : Code
#             fmt code.
#
#         Returns
#         -------
#         int
#             fmt bytesize.
#         """
#
#
# class MessagePatternABC(
#     # ContinuousBytesStoragePatternABC[MessageT, FieldPatternT],
#     Generic[MessageT, FieldPatternT],
# ):
#     """
#     Represent abstract class of pattern for message.
#     """

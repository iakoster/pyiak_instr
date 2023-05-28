"""Private module of ``pyiak_instr.communication.message``."""
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Generic,
    Self,
    TypeVar,
)

from ....core import Code
from ....store.bin.types import (
    BytesStorageABC,
)
from ._struct import (
    MessageFieldStructABC,
    MessageStructGetParser,
    MessageStructHasParser,
    MessageStructABC,
)

if TYPE_CHECKING:
    from ._pattern import MessagePatternABC


__all__ = [
    "MessageABC",
]


AddressT = TypeVar("AddressT")
FieldStructT = TypeVar("FieldStructT", bound=MessageFieldStructABC)
MessageStructT = TypeVar("MessageStructT", bound=MessageStructABC[Any])
MessagePatternT = TypeVar(
    "MessagePatternT", bound="MessagePatternABC[Any, Any]"
)


# todo: clear src and dst?
# todo: get rx and tx class instance
# todo: field parser
class MessageABC(
    BytesStorageABC[FieldStructT, MessageStructT, MessagePatternT],
    Generic[FieldStructT, MessageStructT, MessagePatternT, AddressT],
):
    """
    Represents base class for message.
    """

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

        self._src: AddressT | None = None
        self._dst: AddressT | None = None

    # todo: implement
    def autoupdate_fields(self) -> None:
        """
        Update the content in all fields that support it.
        """

    def split(self) -> Generator[Self, None, None]:
        """
        Split the message into parts over an infinite field.

        Yields
        ------
        Self
            message part.

        Raises
        ------
        ValueError
            if message is empty;
            if message is dynamic and step of dynamic field is not even.
            if address behaviour is undefined.
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

        for dyn_start in range(0, self.bytes_count(dyn_name), dyn_step):
            part = self.__class__(self._s, self._p)  # todo: vulnerability?

            content = b""
            for field in self._s:
                name = field.name

                if name == dyn_name:
                    content += self.content(name)[
                        dyn_start : dyn_start + dyn_step
                    ]

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
    def get(self) -> MessageStructGetParser[MessageStructT, FieldStructT]:
        """
        Returns
        -------
        MessageStructGetParser
            message struct get parser.
        """
        return self._s.get

    @property
    def has(self) -> MessageStructHasParser[MessageStructT, FieldStructT]:
        """
        Returns
        -------
        MessageStructHasParser
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

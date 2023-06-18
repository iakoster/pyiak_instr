"""Private module of ``pyiak_instr.communication.message``."""
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Generic,
    Self,
    TypeVar,
)

from ...core import Code
from ...store.bin import (
    Container as BinContainer,
)
from ._struct import (
    Basic,
    StructGetParser,
    StructHasParser,
    Struct,
)

if TYPE_CHECKING:
    from ._pattern import MessagePattern


__all__ = [
    "Message",
]


AddressT = TypeVar("AddressT")
FieldT = TypeVar("FieldT", bound=Basic)
StructT = TypeVar("StructT", bound=Struct[Any])
PatternT = TypeVar("PatternT", bound="MessagePattern[Any, Any]")


# todo: clear src and dst?
# todo: get rx and tx class instance
# todo: field parser
class Message(
    BinContainer[FieldT, StructT, PatternT],
    Generic[FieldT, StructT, PatternT, AddressT],
):
    """
    Represents base class for message.
    """

    def __init__(
        self,
        struct: StructT,
        pattern: PatternT | None = None,
    ):
        super().__init__(struct, pattern=pattern)
        if self._s.divisible and self.has.address:
            behaviour = self.get.address.behaviour

            if behaviour is not Code.DMA:
                raise TypeError(
                    "invalid address behaviour for divisible message: "
                    f"{behaviour!r}"
                )

        self._src: AddressT | None = None
        self._dst: AddressT | None = None

    def autoupdate_fields(self) -> Self:
        """
        Update the content in all fields that support it.

        Returns
        -------
        Self
            self instance

        Raises
        ------
        ValueError
            if message is empty.
        """
        if self.is_empty():
            raise ValueError("message is empty")

        if self.has.dynamic_length and self.struct.is_dynamic:
            self._autoupdate_dynamic_length_field()

        if self.has.crc:
            self._autoupdate_crc_field()

        return self

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

                elif name == address_name and dyn_start != 0:
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

    def _autoupdate_crc_field(self) -> None:
        """Update content of crc field."""
        crc = self.get.crc
        content = b""

        for field in self.struct:
            if field.name in crc.wo_fields or field is crc:
                continue
            content += self.content(field.name)

        self._change_field_content(
            crc.name, crc.encode(crc.calculate(content), verify=True)
        )

    def _autoupdate_dynamic_length_field(self) -> None:
        """Update content of dynamic length field."""
        dlen = self.get.dynamic_length
        if dlen.behaviour is Code.EXPECTED and self.decode(dlen.name)[0] != 0:
            return

        dyn = self.struct[self.struct.dynamic_field_name]
        self._change_field_content(
            dlen.name,
            dlen.encode(
                dlen.calculate(self.content(dyn.name), dyn.word_bytesize),
                verify=True,
            ),
        )

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
    def get(self) -> StructGetParser[StructT, FieldT]:
        """
        Returns
        -------
        StructGetParser
            message struct get parser.
        """
        return self._s.get

    @property
    def has(self) -> StructHasParser[StructT, FieldT]:
        """
        Returns
        -------
        StructHasParser
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

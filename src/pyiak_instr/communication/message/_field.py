"""Private module of ``pyiak_instr.communication.message`` with field
parsers."""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterable,
    TypeVar,
    Union,
)

from ._struct import (
    MessageFieldStruct,
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DataLengthMessageFieldStruct,
    IdMessageFieldStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
    MessageFieldStructUnionT,
)
from ...core import Code
from ...types.store import BytesFieldABC
from ...exceptions import ContentError

if TYPE_CHECKING:
    from ._message import Message


__all__ = [
    "MessageFieldABC",
    "MessageField",
    "SingleMessageField",
    "StaticMessageField",
    "AddressMessageField",
    "CrcMessageField",
    "DataMessageField",
    "DataLengthMessageField",
    "IdMessageField",
    "OperationMessageField",
    "ResponseMessageField",
    "MessageFieldUnionT",
]


StructT = TypeVar("StructT", bound=MessageFieldStructUnionT)


class MessageFieldABC(BytesFieldABC[StructT], Generic[StructT]):
    """
    Represents abstract class for message field parser.

    Parameters
    ----------
    storage : ContinuousBytesStorage
        storage of fields.
    name : str
        field name.
    struct : BytesFieldStruct
        field struct instance
    """

    def __init__(
        self,
        storage: Message,
        name: str,
        struct: StructT,
    ) -> None:
        super().__init__(name, struct)
        self._storage = storage

    def encode(self, content: int | float | Iterable[int | float]) -> None:
        """
        Encode and write content to the message.

        Parameters
        ----------
        content : int | float | Iterable[int | float]
            content to encoding.
        """
        self._storage.encode(**{self._name: self.content})

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self._storage.content[self._struct.slice_]


class MessageField(MessageFieldABC[MessageFieldStruct]):
    """
    Base message field.
    """


class SingleMessageField(MessageFieldABC[SingleMessageFieldStruct]):
    """
    Message field with one (single) word.
    """


class StaticMessageField(MessageFieldABC[StaticMessageFieldStruct]):
    """
    Message field with static single word (e.g. preamble).
    """


class AddressMessageField(MessageFieldABC[AddressMessageFieldStruct]):
    """
    Message field with address.
    """


class CrcMessageField(MessageFieldABC[CrcMessageFieldStruct]):
    """
    Message field with crc value for message.
    """

    def calculate(self) -> int:
        """
        Calculate an actual crc value of a message.

        Will be calculated for all fields except this field and all fields in
        `wo_fields`.

        Returns
        -------
        int
            actual crc value.
        """
        content = b""
        for name, field in self._storage.items():
            if name not in self._struct.wo_fields and field is not self:
                content += field.content
        return self._struct.calculate(content)

    def update(self) -> None:
        """
        Update crc value in the field.
        """
        self.encode(self.calculate())

    def verify_content(self) -> None:
        """
        Verify content value.

        Raises
        ------
        ContentError
            if field is empty;
            if content not equal to actual crc value.
        """
        if self.is_empty:
            raise ContentError(self, clarification="field is empty")

        if self[0] != self.calculate():
            raise ContentError(self, clarification="invalid crc value")


class DataMessageField(MessageFieldABC[DataMessageFieldStruct]):
    """
    Message field with data.
    """

    def append(self, content: bytes) -> None:
        """
        Append new content to data.

        Parameters
        ----------
        content : bytes
            new content.

        Raises
        ------
        TypeError
            if content not dynamic.
        """
        if self._struct.is_dynamic:
            self.encode(self.content + content)
        else:
            raise TypeError("fails to add new data to a non-dynamic field")


class DataLengthMessageField(MessageFieldABC[DataLengthMessageFieldStruct]):
    """
    Message field with data length value.
    """

    def calculate(self) -> int:
        """
        Calculate actual data length value.

        Returns
        -------
        int
            actual data length.
        """
        if self.struct.behaviour is Code.ACTUAL:
            return self._struct.calculate(
                self._storage["data"].content
            )  # todo: GetParser

        # behaviour is a EXPECTED
        decoded = self[0]
        if decoded == 0:
            return self._struct.calculate(
                self._storage["data"].content
            )  # todo: GetParser
        return decoded  # type: ignore[return-value]

    def update(self) -> None:
        """
        Update data length value in message.
        """
        self.encode(self.calculate())

    def verify_content(self) -> None:
        """
        Verify content value.

        Raises
        ------
        ContentError
            if field is empty;
            if content not equal to actual data length value.
        """
        if self.is_empty:
            raise ContentError(self, clarification="field is empty")

        if self.decode()[0] != self.calculate():
            raise ContentError(
                self, clarification="invalid data length value"
            )


class IdMessageField(MessageFieldABC[IdMessageFieldStruct]):
    """
    Message field with a unique identifier of a particular message.
    """

    # pylint: disable=too-many-return-statements
    def is_equal_to(
        self,
        other: Message
        | IdMessageField
        | int
        | float
        | bytes
        | list[int | float],
    ) -> bool:
        """
        Check that 'other' has the same value as the given instance.

        Parameters
        ----------
        other : Message | IdMessageField | int | float | bytes
            | list[int | float]
            other instance to comparing.

        Returns
        -------
        bool
            True - values are equal, False - they are not.

        Raises
        ------
        TypeError
            if `other` is not comparable type.
        """
        if self.is_empty:
            return False

        if isinstance(other, Message):
            if "id" not in other:  # todo: HasParser
                return False
            return self.content == other["id"].content  # todo: GetParser

        if isinstance(other, IdMessageField):
            other = other.content
        if isinstance(other, bytes):
            return self.content == other

        if isinstance(other, list):
            if len(other) == 0:
                return False
            other = other[0]

        # pylint: disable=isinstance-second-argument-not-valid-type
        if isinstance(other, int | float):
            return self[0] == other

        raise TypeError(f"invalid 'other' type: {other.__class__.__name__}")


class OperationMessageField(MessageFieldABC[OperationMessageFieldStruct]):
    """
    Message field with operation (e.g. read).
    """

    def desc(self) -> Code:
        """
        Operation description.

        If code not represented - returns UNDEFINED.

        Returns
        -------
        Code
            operation code.
        """
        return self._struct.desc(self[0])  # type: ignore[arg-type]


class ResponseMessageField(
    MessageFieldABC[ResponseMessageFieldStruct],
):
    """
    Message field with response.
    """

    def desc(self) -> Code:
        """
        Response description.

        If code not represented - returns UNDEFINED.

        Returns
        -------
        Code
            Response code.
        """
        return self._struct.desc(self[0])  # type: ignore[arg-type]


MessageFieldUnionT = Union[
    MessageField,
    SingleMessageField,
    StaticMessageField,
    AddressMessageField,
    CrcMessageField,
    DataMessageField,
    DataLengthMessageField,
    IdMessageField,
    OperationMessageField,
    ResponseMessageField,
]

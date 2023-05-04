"""Private module of ``pyiak_instr.communication.message`` with field
parsers."""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Union,
    cast,
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
from ...types.communication import MessageFieldABC
from ...exceptions import ContentError

if TYPE_CHECKING:
    from ._message import Message  # noqa: F401


__all__ = [
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


class MessageField(MessageFieldABC["Message", MessageFieldStruct]):
    """
    Base message field.
    """


class SingleMessageField(
    MessageFieldABC["Message", SingleMessageFieldStruct]
):
    """
    Message field with one (single) word.
    """


class StaticMessageField(
    MessageFieldABC["Message", StaticMessageFieldStruct]
):
    """
    Message field with static single word (e.g. preamble).
    """


class AddressMessageField(
    MessageFieldABC["Message", AddressMessageFieldStruct]
):
    """
    Message field with address.
    """


class CrcMessageField(MessageFieldABC["Message", CrcMessageFieldStruct]):
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


class DataMessageField(MessageFieldABC["Message", DataMessageFieldStruct]):
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
        if not self.struct.is_dynamic:
            raise TypeError("fails to add new data to a non-dynamic field")
        self.encode(self.content + content)


class DataLengthMessageField(
    MessageFieldABC["Message", DataLengthMessageFieldStruct]
):
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
            data_field = self._storage.get.data
            return self._struct.calculate(
                data_field.content, data_field.struct.fmt
            )

        # behaviour is a EXPECTED
        decoded = self[0]
        if decoded == 0:
            data_field = self._storage.get.data
            return self._struct.calculate(
                data_field.content, data_field.struct.fmt
            )
        return cast(int, decoded)

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


class IdMessageField(MessageFieldABC["Message", IdMessageFieldStruct]):
    """
    Message field with a unique identifier of a particular message.
    """

    def is_equal_to(
        self,
        other: IdMessageField | int | float | bytes | list[int | float],
    ) -> bool:
        """
        Check that 'other' has the same value as the given instance.

        Parameters
        ----------
        other : IdMessageField | int | float | bytes | list[int | float]
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

        if isinstance(other, IdMessageField):
            other = other.content
        if isinstance(other, bytes):
            return self.content == other

        if isinstance(other, list):
            if len(other) != 1:
                return False
            other = other[0]

        if isinstance(other, int | float):
            return self[0] == other

        raise TypeError(f"invalid 'other' type: {other.__class__.__name__}")


class OperationMessageField(
    MessageFieldABC["Message", OperationMessageFieldStruct]
):
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
        return self._struct.desc(cast(int, self[0]))


class ResponseMessageField(
    MessageFieldABC["Message", ResponseMessageFieldStruct],
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
        return self._struct.desc(cast(int, self[0]))


MessageFieldUnionT = Union[  # pylint: disable=invalid-name
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

"""Private module of ``pyiak_instr.communication.message`` with field
parsers."""
from __future__ import annotations
from typing import Any, Generator, Self

import numpy.typing as npt

from ._field import (
    MessageFieldParameters,
    SingleMessageFieldParameters,
    StaticMessageFieldParameters,
    AddressMessageFieldParameters,
    CrcMessageFieldParameters,
    DataMessageFieldParameters,
    DataLengthMessageFieldParameters,
    IdMessageFieldParameters,
    OperationMessageFieldParameters,
    ResponseMessageFieldParameters,
    MessageFieldUnionT,
)
from ...store import BytesField, ContinuousBytesStorage
from ...core import Code
from ...typing import BytesFieldABC

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
]


# todo: __str__
class Message(ContinuousBytesStorage):
    """
    Represents message for communication.

    Parameters
    ----------
    name: str, default='std'
        name of storage.
    splittable: bool, default=False
        shows that the message can be divided by the data.
    slice_length: int, default=1024
        max length of the data in one slice.
    **fields: BytesField
        fields of the storage. The kwarg Key is used as the field name.
    """

    def __init__(
        self,
        name: str = "std",
        splittable: bool = False,
        slice_length: int = 1024,  # todo: length must in bytes
        **fields: MessageFieldUnionT,
    ) -> None:
        if "data" not in fields:
            raise KeyError("field with name 'data' required")
        super().__init__(name, **fields)
        self._splittable = splittable
        self._slice_length = slice_length

        self._src, self._dst = None, None
        # self._field_types: dict[type[MessageFieldUnionT], str] = {}
        # for field in self:
        #     field_class = field.fld.__class__
        #     if field_class not in self._field_types:
        #         self._field_types[field_class] = field.name

    def set_src_dst(self, src: Any, dst: Any) -> Self:
        """
        Set src and dst addresses.

        Addresses may differ depending on the type of connection used.

        Parameters
        ----------
        src: Any
            source address.
        dst: Any
            destination address.

        Returns
        -------
        FieldMessage
            object message instance.
        """
        self._src, self._dst = src, dst
        return self

    def split(self) -> Generator[Message, None, None]:
        """
        Split data field on slices.

        Raises
        ------
        NotImplementedError
            placeholder
        """
        raise NotImplementedError()

    @property
    def dst(self) -> Any:
        """
        Returns
        -------
        Any
            destination address.
        """
        return self._dst

    @dst.setter
    def dst(self, destination: Any) -> None:
        """
        Set destination address.

        Parameters
        ----------
        destination: Any
            destination address.
        """
        self._dst = destination

    @property
    def slice_length(self) -> int:
        """
        If splittable is True that this attribute can be used.

        Returns
        -------
        int
            max length of the data field in message for sending.
        """
        return self._slice_length

    @property
    def splittable(self) -> bool:
        """
        Indicates that the message can be splited.

        Returns
        -------
        bool
            pointer to the possibility of separation
        """
        return self._splittable

    @property
    def src(self) -> Any:
        """
        Returns
        -------
        Any
            source address.
        """
        return self._src

    @src.setter
    def src(self, source: Any) -> None:
        """
        Set source address.

        Parameters
        ----------
        source: Any
            source address.
        """
        self._src = source


class MessageField(
    BytesField, BytesFieldABC[Message, MessageFieldParameters]
):
    """
    Represents parser for work with message field content.
    """


class SingleMessageField(
    MessageField, BytesFieldABC[Message, SingleMessageFieldParameters]
):
    """
    Represents parser for work with single message field content.
    """


class StaticMessageField(
    MessageField, BytesFieldABC[Message, StaticMessageFieldParameters]
):
    """
    Represents parser for work with static message field content.
    """


class AddressMessageField(
    SingleMessageField, BytesFieldABC[Message, AddressMessageFieldParameters]
):
    """
    Represents parser for work with crc message field content.
    """


class CrcMessageField(
    SingleMessageField, BytesFieldABC[Message, CrcMessageFieldParameters]
):
    """
    Represents parser for work with crc message field content.
    """

    def calculate(self) -> int:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        # content = b""
        # for field in msg:
        #     if field.name not in self.fld.wo_fields and field is not self:
        #         content += field.content
        # return self.fld.algorithm(content)
        raise NotImplementedError()

    def update(self) -> None:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()


class DataMessageField(
    MessageFieldParameters, BytesFieldABC[Message, DataMessageFieldParameters]
):
    """
    Represents parser for work with data message field content.
    """

    def append(self, content: npt.ArrayLike) -> None:
        """
        append new content.

        Parameters
        ----------
        content : ArrayLike
            new content.

        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()


class DataLengthMessageField(
    SingleMessageField,
    BytesFieldABC[Message, DataLengthMessageFieldParameters],
):
    """
    Represents parser for work with data length message field content.
    """

    def calculate(self) -> None:
        """
        Raises
        -------
        NotImplementedError
            placeholder
        """
        raise NotImplementedError()

    def update(self) -> None:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()


class IdMessageField(
    SingleMessageField, BytesFieldABC[Message, IdMessageFieldParameters]
):
    """
    Represents parser for work with id message field content.
    """

    def compare(self) -> None:
        """
        Returns
        -------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()


class OperationMessageField(
    SingleMessageField,
    BytesFieldABC[Message, OperationMessageFieldParameters],
):
    """
    Represents parser for work with operation message field content.
    """

    def calculate(self) -> None:
        """
        Raises
        -------
        NotImplementedError
            placeholder
        """
        raise NotImplementedError()

    def update(self) -> None:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()


class ResponseMessageField(
    SingleMessageField, BytesFieldABC[Message, ResponseMessageFieldParameters]
):
    """
    Represents parser for work with response message field content.
    """

    @property
    def code(self) -> Code:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

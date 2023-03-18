"""Private module of ``pyiak_instr.communication.message`` with field
parsers."""
from __future__ import annotations
from typing import Any, Generator, Self

import numpy.typing as npt

from ._field import (
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
    MessageFieldUnionT,
)
from ...store import BytesFieldParser, ContinuousBytesStorage
from ...core import Code

__all__ = [
    "MessageFieldParser",
    "SingleMessageFieldParser",
    "StaticMessageFieldParser",
    "AddressMessageFieldParser",
    "CrcMessageFieldParser",
    "DataMessageFieldParser",
    "DataLengthMessageFieldParser",
    "IdMessageFieldParser",
    "OperationMessageFieldParser",
    "ResponseMessageFieldParser",
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


# todo: all parsers via generic
class MessageFieldParser(BytesFieldParser):
    """
    Represents parser for work with message field content.
    """

    _s: Message
    _f: MessageField

    @property
    def fld(self) -> MessageField:
        """
        Returns
        -------
        MessageField
            field instance
        """
        return self._f


class SingleMessageFieldParser(MessageFieldParser):
    """
    Represents parser for work with single message field content.
    """

    _f: SingleMessageField

    @property
    def fld(self) -> SingleMessageField:
        """
        Returns
        -------
        SingleMessageField
            field instance
        """
        return self._f


class StaticMessageFieldParser(MessageFieldParser):
    """
    Represents parser for work with static message field content.
    """

    _f: StaticMessageField

    @property
    def fld(self) -> StaticMessageField:
        """
        Returns
        -------
        StaticMessageField
            field instance.
        """
        return self._f


class AddressMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with crc message field content.
    """

    _f: AddressMessageField

    @property
    def fld(self) -> AddressMessageField:
        """
        Returns
        -------
        AddressMessageField
            field instance.
        """
        return self._f


class CrcMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with crc message field content.
    """

    _f: CrcMessageField

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

    @property
    def fld(self) -> CrcMessageField:
        """
        Returns
        -------
        CrcMessageField
            field instance.
        """
        return self._f


class DataMessageFieldParser(MessageField):
    """
    Represents parser for work with data message field content.
    """

    _f: DataMessageField

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

    @property
    def fld(self) -> DataMessageField:
        """
        Returns
        -------
        DataMessageField
            field instance.
        """
        return self._f


class DataLengthMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with data length message field content.
    """

    _f: DataLengthMessageField

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

    @property
    def fld(self) -> DataLengthMessageField:
        """
        Returns
        -------
        DataLengthMessageField
            field instance.
        """
        return self._f


class IdMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with id message field content.
    """

    _f: IdMessageField

    def compare(self) -> None:
        """
        Returns
        -------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> IdMessageField:
        """
        Returns
        -------
        IdMessageField
            field instance.
        """
        return self._f


class OperationMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with operation message field content.
    """

    _f: OperationMessageField

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

    @property
    def fld(self) -> OperationMessageField:
        """
        Returns
        -------
        OperationMessageField
            self instance.
        """
        return self._f


class ResponseMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with response message field content.
    """

    _f: ResponseMessageField

    @property
    def code(self) -> Code:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> ResponseMessageField:
        """
        Returns
        -------
        ResponseMessageField
            self instance.
        """
        return self._f

"""Private module of ``pyiak_instr.communication.message`` with field
parsers."""
from __future__ import annotations
from typing import Any, Generator, Self, Union

import numpy.typing as npt

from ._field import (
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
    MessageFieldPattern,
)
from ...store import BytesField, ContinuousBytesStorage, BytesStoragePattern
from ...core import Code
from ...typing import BytesFieldABC, PatternStorageABC

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


# todo: fix return typing (how? i don't known)
class MessageGetParser:
    """
    Represents parser to get the field type from message.

    Parameters
    ----------
    message: Message
        message instance.
    types: dict[type[FieldType], str]
        dictionary of field types in the message.
    """

    def __init__(
        self,
        message: Message,
        types: dict[type[MessageFieldStructUnionT], str],
    ):
        self._msg, self._types = message, types

    @property
    def basic(self) -> MessageField:
        """
        Returns
        -------
        MessageField
            field instance.
        """
        return self(MessageFieldStruct)  # type: ignore[return-value]

    @property
    def single(self) -> SingleMessageField:
        """
        Returns
        -------
        SingleMessageField
            field instance.
        """
        return self(SingleMessageFieldStruct)  # type: ignore[return-value]

    @property
    def static(self) -> StaticMessageField:
        """
        Returns
        -------
        StaticMessageField
            field instance.
        """
        return self(StaticMessageFieldStruct)  # type: ignore[return-value]

    @property
    def address(self) -> AddressMessageField:
        """
        Returns
        -------
        AddressMessageField
            field instance.
        """
        return self(AddressMessageFieldStruct)  # type: ignore[return-value]

    @property
    def crc(self) -> CrcMessageField:
        """
        Returns
        -------
        CrcMessageField
            field instance.
        """
        return self(CrcMessageFieldStruct)  # type: ignore[return-value]

    @property
    def data(self) -> DataMessageField:
        """
        Returns
        -------
        DataMessageField
            field instance.
        """
        return self(DataMessageFieldStruct)  # type: ignore[return-value]

    @property
    def data_length(self) -> DataLengthMessageField:
        """
        Returns
        -------
        DataLengthMessageField
            field instance.
        """
        return self(
            DataLengthMessageFieldStruct
        )  # type: ignore[return-value]

    # pylint: disable=invalid-name
    @property
    def id(self) -> IdMessageField:
        """
        Returns
        -------
        IdMessageField
            field instance.
        """
        return self(IdMessageFieldStruct)  # type: ignore[return-value]

    @property
    def operation(self) -> OperationMessageField:
        """
        Returns
        -------
        OperationMessageField
            field instance.
        """
        return self(OperationMessageFieldStruct)  # type: ignore[return-value]

    @property
    def response(self) -> ResponseMessageField:
        """
        Returns
        -------
        ResponseMessageField
            field instance.
        """
        return self(ResponseMessageFieldStruct)  # type: ignore[return-value]

    def __call__(
        self, type_: type[MessageFieldStructUnionT]
    ) -> MessageFieldUnionT:
        """
        Get first field with specified type.

        Parameters
        ----------
        type_: type[MessageFieldParametersUnionT]

        Returns
        -------
        MessageFieldUnionT
            field if specified type.

        Raises
        ------
        TypeError
            if type not found in fields list.
        """
        if type_ not in self._types:
            raise TypeError("there is no field with type %s" % type_.__name__)
        return self._msg[self._types[type_]]  # type: ignore[return-value]


class MessageHasParser:
    """
    Represents parser to check the field class exists in the message.

    Parameters
    ----------
    types: set[type[MessageFieldParametersUnionT]]
        set of field types in the message.
    """

    def __init__(
        self,
        types: set[type[MessageFieldStructUnionT]],
    ) -> None:
        self._types = types

    @property
    def basic(self) -> bool:
        """
        Returns
        -------
        bool
            True -- basic field exists in message.
        """
        return self(MessageFieldStruct)

    @property
    def single(self) -> bool:
        """
        Returns
        -------
        bool
            True -- single field exists in message.
        """
        return self(SingleMessageFieldStruct)

    @property
    def static(self) -> bool:
        """
        Returns
        -------
        bool
            True -- static field exists in message.
        """
        return self(StaticMessageFieldStruct)

    @property
    def address(self) -> bool:
        """
        Returns
        -------
        bool
            True -- address field exists in message.
        """
        return self(AddressMessageFieldStruct)

    @property
    def crc(self) -> bool:
        """
        Returns
        -------
        bool
            True -- crc field exists in message.
        """
        return self(CrcMessageFieldStruct)

    @property
    def data(self) -> bool:
        """
        Returns
        -------
        bool
            True -- data field exists in message.
        """
        return self(DataMessageFieldStruct)

    @property
    def data_length(self) -> bool:
        """
        Returns
        -------
        bool
            True -- data length field exists in message.
        """
        return self(DataLengthMessageFieldStruct)

    # pylint: disable=invalid-name
    @property
    def id(self) -> bool:
        """
        Returns
        -------
        bool
            True -- id field exists in message.
        """
        return self(IdMessageFieldStruct)

    @property
    def operation(self) -> bool:
        """
        Returns
        -------
        bool
            True -- operation field exists in message.
        """
        return self(OperationMessageFieldStruct)

    @property
    def response(self) -> bool:
        """
        Returns
        -------
        bool
            True -- response field exists in message.
        """
        return self(ResponseMessageFieldStruct)

    def __call__(self, type_: type[MessageFieldStructUnionT]) -> bool:
        """
        Check that message has field of specified type.

        Parameters
        ----------
        type_: type[MessageFieldParametersUnionT]
            the type of field whose existence is to be checked.

        Returns
        -------
        bool
            True - if field type exists, False - not.
        """
        return type_ in self._types


# todo: __str__
# todo: fix typing
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
        **fields: MessageFieldStructUnionT,
    ) -> None:
        if "data" not in fields:
            raise KeyError("field with name 'data' required")
        super().__init__(name, **fields)
        self._splittable = splittable
        self._slice_length = slice_length

        self._src, self._dst = None, None
        self._field_types: dict[type[MessageFieldStructUnionT], str] = {}
        for f_name, parameters in fields.items():
            class_ = parameters.__class__
            if class_ not in self._field_types:
                self._field_types[class_] = f_name

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
    def get(self) -> MessageGetParser:
        """
        Returns
        -------
        MessageGetParser
            get parser instance.
        """
        return MessageGetParser(self, self._field_types)

    @property
    def has(self) -> MessageHasParser:
        """
        Returns
        -------
        MessageHasParser
            has parser instance.
        """
        return MessageHasParser(set(self._field_types))

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


class MessageField(BytesField, BytesFieldABC[Message, MessageFieldStruct]):
    """
    Represents parser for work with message field content.
    """


class SingleMessageField(
    MessageField, BytesFieldABC[Message, SingleMessageFieldStruct]
):
    """
    Represents parser for work with single message field content.
    """


class StaticMessageField(
    MessageField, BytesFieldABC[Message, StaticMessageFieldStruct]
):
    """
    Represents parser for work with static message field content.
    """


class AddressMessageField(
    SingleMessageField, BytesFieldABC[Message, AddressMessageFieldStruct]
):
    """
    Represents parser for work with crc message field content.
    """


class CrcMessageField(
    SingleMessageField, BytesFieldABC[Message, CrcMessageFieldStruct]
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
    MessageFieldStruct, BytesFieldABC[Message, DataMessageFieldStruct]
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
    BytesFieldABC[Message, DataLengthMessageFieldStruct],
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
    SingleMessageField, BytesFieldABC[Message, IdMessageFieldStruct]
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
    BytesFieldABC[Message, OperationMessageFieldStruct],
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
    SingleMessageField, BytesFieldABC[Message, ResponseMessageFieldStruct]
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


# todo: implement storage only for one
# todo: fix typing (how?)
class MessagePattern(
    BytesStoragePattern, PatternStorageABC[Message, MessageFieldPattern]
):
    """
    Represents class which storage common parameters for message.

    Parameters
    ----------
    name: str
        name of message format.
    splittable: bool, default=False
        shows that the message can be divided by the data.
    slice_length: int, default=1024
        max length of the data in one slice.
    """

    _target_options = {}
    _target_default = Message  # type: ignore[assignment]

    def __init__(
        self, name: str, splittable: bool = False, slice_length: int = 1024
    ):
        super().__init__(
            "continuous",
            name,
            splittable=splittable,
            slice_length=slice_length,
        )

    def get(self, changes_allowed: bool = False, **additions: Any) -> Message:
        """
        Get initialized message.

        Parameters
        ----------
        changes_allowed: bool, default = False
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection, otherwise the
            `additions` take precedence.
        **additions: Any
            additional initialization parameters. Those keys that are
            separated by "__" will be defined as parameters for other
            patterns target, otherwise for the storage target.

        Returns
        -------
        ContinuousBytesStorage
            initialized storage.

        Raises
        ------
        AssertionError
            if in some reason typename is invalid.
        NotConfiguredYet
            if patterns list is empty.
        """
        return super().get(  # type: ignore[return-value]
            changes_allowed=changes_allowed, **additions
        )

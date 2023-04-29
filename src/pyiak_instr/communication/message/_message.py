"""Private module of ``pyiak_instr.communication.message`` with message
classes."""
from __future__ import annotations

from ...types.communication import (
    MessageABC,
    MessageGetParserABC,
    MessageHasParserABC,
    MessagePatternABC,
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
    MessageFieldPattern,
)
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


class MessageGetParser(MessageGetParserABC["Message", MessageFieldUnionT]):
    """
    Parser to get the field from message by it type.
    """

    @property
    def basic(self) -> MessageField:
        """
        Returns
        -------
        MessageField
            field instance.
        """
        return self(MessageField)

    @property
    def single(self) -> SingleMessageField:
        """
        Returns
        -------
        SingleMessageField
            field instance.
        """
        return self(SingleMessageField)

    @property
    def static(self) -> StaticMessageField:
        """
        Returns
        -------
        StaticMessageField
            field instance.
        """
        return self(StaticMessageField)

    @property
    def address(self) -> AddressMessageField:
        """
        Returns
        -------
        AddressMessageField
            field instance.
        """
        return self(AddressMessageField)

    @property
    def crc(self) -> CrcMessageField:
        """
        Returns
        -------
        CrcMessageField
            field instance.
        """
        return self(CrcMessageField)

    @property
    def data(self) -> DataMessageField:
        """
        Returns
        -------
        DataMessageField
            field instance.
        """
        return self(DataMessageField)

    @property
    def data_length(self) -> DataLengthMessageField:
        """
        Returns
        -------
        DataLengthMessageField
            field instance.
        """
        return self(DataLengthMessageField)

    @property
    def id_(self) -> IdMessageField:
        """
        Returns
        -------
        IdMessageField
            field instance.
        """
        return self(IdMessageField)

    @property
    def operation(self) -> OperationMessageField:
        """
        Returns
        -------
        OperationMessageField
            field instance.
        """
        return self(OperationMessageField)

    @property
    def response(self) -> ResponseMessageField:
        """
        Returns
        -------
        ResponseMessageField
            field instance.
        """
        return self(ResponseMessageField)


class MessageHasParser(MessageHasParserABC[MessageFieldUnionT]):
    """
    Parser to check the field class exists in the message.
    """

    @property
    def basic(self) -> bool:
        """
        Returns
        -------
        bool
            True -- basic field exists in message.
        """
        return self(MessageField)

    @property
    def single(self) -> bool:
        """
        Returns
        -------
        bool
            True -- single field exists in message.
        """
        return self(SingleMessageField)

    @property
    def static(self) -> bool:
        """
        Returns
        -------
        bool
            True -- static field exists in message.
        """
        return self(StaticMessageField)

    @property
    def address(self) -> bool:
        """
        Returns
        -------
        bool
            True -- address field exists in message.
        """
        return self(AddressMessageField)

    @property
    def crc(self) -> bool:
        """
        Returns
        -------
        bool
            True -- crc field exists in message.
        """
        return self(CrcMessageField)

    @property
    def data(self) -> bool:
        """
        Returns
        -------
        bool
            True -- data field exists in message.
        """
        return self(DataMessageField)

    @property
    def data_length(self) -> bool:
        """
        Returns
        -------
        bool
            True -- data length field exists in message.
        """
        return self(DataLengthMessageField)

    @property
    def id_(self) -> bool:
        """
        Returns
        -------
        bool
            True -- id field exists in message.
        """
        return self(IdMessageField)

    @property
    def operation(self) -> bool:
        """
        Returns
        -------
        bool
            True -- operation field exists in message.
        """
        return self(OperationMessageField)

    @property
    def response(self) -> bool:
        """
        Returns
        -------
        bool
            True -- response field exists in message.
        """
        return self(ResponseMessageField)


# todo: __str__
class Message(
    MessageABC[
        MessageFieldUnionT,
        MessageFieldStructUnionT,
        MessageGetParser,
        MessageHasParser,
    ]
):
    """
    Message for communication between devices.
    """

    _get_parser = MessageGetParser
    _has_parser = MessageHasParser
    _struct_field = {
        MessageFieldStruct: MessageField,
        SingleMessageFieldStruct: SingleMessageField,
        StaticMessageFieldStruct: StaticMessageField,
        AddressMessageFieldStruct: AddressMessageField,
        CrcMessageFieldStruct: CrcMessageField,
        DataMessageFieldStruct: DataMessageField,
        DataLengthMessageFieldStruct: DataLengthMessageField,
        IdMessageFieldStruct: IdMessageField,
        OperationMessageFieldStruct: OperationMessageField,
        ResponseMessageFieldStruct: ResponseMessageField,
    }


class MessagePattern(MessagePatternABC[Message, MessageFieldPattern]):
    """
    Pattern for message class
    """
#
#
# class MessagePattern(
#     BytesStoragePattern, PatternStorageABC[Message, MessageFieldPattern]
# ):
#     """
#     Represents class which storage common parameters for message.
#
#     Parameters
#     ----------
#     name: str
#         name of message format.
#     splittable: bool, default=False
#         shows that the message can be divided by the data.
#     slice_length: int, default=1024
#         max length of the data in one slice.
#     """
#
#     _target_options = {}
#     _target_default = Message  # type: ignore[assignment]
#
#     def __init__(
#         self, name: str, splittable: bool = False, slice_length: int = 1024
#     ):
#         super().__init__(
#             "continuous",
#             name,
#             splittable=splittable,
#             slice_length=slice_length,
#         )
#
#     def get(
#     self, changes_allowed: bool = False, **additions: Any) -> Message:
#         """
#         Get initialized message.
#
#         Parameters
#         ----------
#         changes_allowed: bool, default = False
#             allows situations where keys from the pattern overlap with
#             kwargs.
#             If False, it causes an error on intersection, otherwise the
#             `additions` take precedence.
#         **additions: Any
#             additional initialization parameters. Those keys that are
#             separated by "__" will be defined as parameters for other
#             patterns target, otherwise for the storage target.
#
#         Returns
#         -------
#         ContinuousBytesStorage
#             initialized storage.
#
#         Raises
#         ------
#         AssertionError
#             if in some reason typename is invalid.
#         NotConfiguredYet
#             if patterns list is empty.
#         """
#         return super().get(  # type: ignore[return-value]
#             changes_allowed=changes_allowed, **additions
#         )

# """Private module of ``pyiak_instr.communication.message`` with message
# classes."""
# from __future__ import annotations
# from typing import Any, Generator, Self
#
# from ...core import Code
# from ...types.communication import (
#     MessageABC,
#     MessageGetParserABC,
#     MessageHasParserABC,
#     MessagePatternABC,
# )
# from ._struct import (
#     MessageFieldStruct,
#     SingleMessageFieldStruct,
#     StaticMessageFieldStruct,
#     AddressMessageFieldStruct,
#     CrcMessageFieldStruct,
#     DataMessageFieldStruct,
#     DataLengthMessageFieldStruct,
#     IdMessageFieldStruct,
#     OperationMessageFieldStruct,
#     ResponseMessageFieldStruct,
#     MessageFieldStructUnionT,
#     MessageFieldPattern,
# )
# from ._field import (
#     MessageField,
#     SingleMessageField,
#     StaticMessageField,
#     AddressMessageField,
#     CrcMessageField,
#     DataMessageField,
#     DataLengthMessageField,
#     IdMessageField,
#     OperationMessageField,
#     ResponseMessageField,
#     MessageFieldUnionT,
# )
#
#
# __all__ = ["Message", "MessagePattern"]
#
#
# class MessageGetParser(MessageGetParserABC["Message", MessageFieldUnionT]):
#     """
#     Parser to get the field from message by it type.
#     """
#
#     @property
#     def basic(self) -> MessageField:
#         """
#         Returns
#         -------
#         MessageField
#             field instance.
#         """
#         return self(MessageField)
#
#     @property
#     def single(self) -> SingleMessageField:
#         """
#         Returns
#         -------
#         SingleMessageField
#             field instance.
#         """
#         return self(SingleMessageField)
#
#     @property
#     def static(self) -> StaticMessageField:
#         """
#         Returns
#         -------
#         StaticMessageField
#             field instance.
#         """
#         return self(StaticMessageField)
#
#     @property
#     def address(self) -> AddressMessageField:
#         """
#         Returns
#         -------
#         AddressMessageField
#             field instance.
#         """
#         return self(AddressMessageField)
#
#     @property
#     def crc(self) -> CrcMessageField:
#         """
#         Returns
#         -------
#         CrcMessageField
#             field instance.
#         """
#         return self(CrcMessageField)
#
#     @property
#     def data(self) -> DataMessageField:
#         """
#         Returns
#         -------
#         DataMessageField
#             field instance.
#         """
#         return self(DataMessageField)
#
#     @property
#     def data_length(self) -> DataLengthMessageField:
#         """
#         Returns
#         -------
#         DataLengthMessageField
#             field instance.
#         """
#         return self(DataLengthMessageField)
#
#     @property
#     def id_(self) -> IdMessageField:
#         """
#         Returns
#         -------
#         IdMessageField
#             field instance.
#         """
#         return self(IdMessageField)
#
#     @property
#     def operation(self) -> OperationMessageField:
#         """
#         Returns
#         -------
#         OperationMessageField
#             field instance.
#         """
#         return self(OperationMessageField)
#
#     @property
#     def response(self) -> ResponseMessageField:
#         """
#         Returns
#         -------
#         ResponseMessageField
#             field instance.
#         """
#         return self(ResponseMessageField)
#
#
# class MessageHasParser(MessageHasParserABC[MessageFieldUnionT]):
#     """
#     Parser to check the field class exists in the message.
#     """
#
#     @property
#     def basic(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- basic field exists in message.
#         """
#         return self(MessageField)
#
#     @property
#     def single(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- single field exists in message.
#         """
#         return self(SingleMessageField)
#
#     @property
#     def static(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- static field exists in message.
#         """
#         return self(StaticMessageField)
#
#     @property
#     def address(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- address field exists in message.
#         """
#         return self(AddressMessageField)
#
#     @property
#     def crc(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- crc field exists in message.
#         """
#         return self(CrcMessageField)
#
#     @property
#     def data(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- data field exists in message.
#         """
#         return self(DataMessageField)
#
#     @property
#     def data_length(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- data length field exists in message.
#         """
#         return self(DataLengthMessageField)
#
#     @property
#     def id_(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- id field exists in message.
#         """
#         return self(IdMessageField)
#
#     @property
#     def operation(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- operation field exists in message.
#         """
#         return self(OperationMessageField)
#
#     @property
#     def response(self) -> bool:
#         """
#         Returns
#         -------
#         bool
#             True -- response field exists in message.
#         """
#         return self(ResponseMessageField)
#
#
# # todo: __str__
# class Message(
#     MessageABC[
#         "MessagePattern",
#         MessageFieldUnionT,
#         MessageFieldStructUnionT,
#         MessageGetParser,
#         MessageHasParser,
#         Any,  # todo: to Generic?
#     ]
# ):
#     """
#     Message for communication between devices.
#     """
#
#     _get_parser = MessageGetParser
#     _has_parser = MessageHasParser
#     _struct_field = {
#         MessageFieldStruct: MessageField,
#         SingleMessageFieldStruct: SingleMessageField,
#         StaticMessageFieldStruct: StaticMessageField,
#         AddressMessageFieldStruct: AddressMessageField,
#         CrcMessageFieldStruct: CrcMessageField,
#         DataMessageFieldStruct: DataMessageField,
#         DataLengthMessageFieldStruct: DataLengthMessageField,
#         IdMessageFieldStruct: IdMessageField,
#         OperationMessageFieldStruct: OperationMessageField,
#         ResponseMessageFieldStruct: ResponseMessageField,
#     }
#
#     def split(self) -> Generator[Self, None, None]:
#         if len(self) == 0:
#             raise TypeError("message is empty")
#
#         if not self._div or len(self._c) < self._mtu:
#             yield self
#             return
#
#         if not self.is_dynamic:
#             # todo: warning
#             yield self
#             return
#
#         if (
#             self.has.address
#             and self.get.address.struct.behaviour is not Code.DMA
#         ):
#             # todo: warning
#             yield self
#             return
#
#         f_dyn = self[self._dyn_field]
#         dyn_size = self._mtu - self.minimum_size
#         if f_dyn.bytes_count > 0:
#             if self.has.address:
#                 address = self.get.address[0]
#             else:
#                 address = -1
#
#             part = self._p.get()
#
#         elif self.has.data_length and self.get.data_length[0] > 0:
#             ...
#
#         else:
#             yield self
#
#
# class MessagePattern(MessagePatternABC[Message, MessageFieldPattern]):
#     """
#     Pattern for message class
#     """
#
#     _options = {"basic": Message}

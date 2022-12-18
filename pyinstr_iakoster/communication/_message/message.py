from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Generator,
    Set,
    ClassVar,
    Any,
    overload,
)

import numpy as np
import numpy.typing as npt

from .field import (
    ContentType,
    Field,
    SingleField,
    StaticField,
    AddressField,
    CrcField,
    DataField,
    DataLengthField,
    OperationField,
    ResponseField,
    FieldSetter,
    FieldType,
)
from ...exceptions import (
    MessageContentError,
    NotConfiguredMessageError,
)
from ...core import Code


__all__ = [
    "MessageType",
    "BaseMessage",
    "BytesMessage",
    "FieldMessage",
    "StrongFieldMessage",
    "MessageSetter",
    "MessageContentError",
    "NotConfiguredMessageError",
]

# todo: add tests


class BaseMessage(object):

    def __init__(
            self,
            mf_name: str = "std",
            splittable: bool = False,
            slice_length: int = 1024,
    ):  # todo: test to default values
        self._mf_name = mf_name
        self._splittable = splittable
        self._slice_length = slice_length

        self._src, self._dst = None, None

    def set(self, *args, **kwargs) -> BaseMessage:
        """
        Set message content.

        Returns
        -------
        BaseMessage
            self class instance.
        """
        raise NotImplementedError()

    def split(self) -> Generator[BaseMessage, None, None]:
        """
        Split content of message by slice length.

        Yields
        ------
        BaseMessage
            message part.
        """
        raise NotImplementedError()

    def in_bytes(self) -> bytes:
        """
        Returns
        -------
        bytes
            message content in bytes.
        """
        raise NotImplementedError()

    def unpack(self) -> npt.NDArray:
        """
        Returns
        -------
        numpy.typing.NDArray
            unpacked message content.
        """
        raise NotImplementedError()

    def _content_repr(self) -> str:
        """Returns string representation of the content."""
        raise NotImplementedError()

    def __add__(self, other: BaseMessage | bytes) -> BaseMessage:
        """
        add new data to the Message.

        Parameters
        ----------
        other: BaseMessage or bytes
            additional data or message.

        Returns
        -------
        BaseMessage
            self instance.
        """
        raise NotImplementedError()

    def __getitem__(self, item: Any) -> Any:
        """
        Returns specified item of the message.
        """
        raise NotImplementedError()

    def __iter__(self) -> Generator[Any, None, None]:
        """Iteration by message."""
        raise NotImplementedError()

    def clear_src_dst(self) -> None:
        """Set `src` and `dst` to None."""
        self._dst, self._src = None, None

    def get_instance(self) -> BaseMessage:
        """
        Get the same class as the current object, initialized with
        the same arguments, but with empty content.

        Returns
        -------
        BaseMessage
            new class instance.
        """
        return self.__class__(
            mf_name=self._mf_name,
            splittable=self._splittable,
            slice_length=self._slice_length,
        )

    def get_setter(self) -> MessageSetter:
        """
        Get setter of the message instance.

        Returns
        -------
        MessageSetter
            setter for this message instance.
        """
        if self.__class__ is BaseMessage:
            raise ValueError("BaseMessage not supported by setter")

        types = MessageSetter.MESSAGE_TYPES
        message_type = list(types.keys())[
            list(types.values()).index(self.__class__)
        ]
        return MessageSetter(
            message_type=message_type,
            mf_name=self._mf_name,
            splittable=self._splittable,
            slice_length=self.slice_length,
        )

    def set_src_dst(self, src: Any, dst: Any) -> BaseMessage:
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
    def dst(self, dst_: Any) -> None:
        """
        Set destination address.

        Parameters
        ----------
        dst_: Any
            destination address.
        """
        self._dst = dst_

    @property
    def mf_name(self) -> str:
        """
        Returns
        -------
        str
            name of the message format.
        """
        return self._mf_name

    @property
    def slice_length(self) -> int:
        """
        If splittable is True that this attribute can be used.

        Returns
        -------
        int
            max length of the data field in message for sending.

        See Also
        --------
        pyinstr_iakoster.communication.Message.split: method for splitting
            message for sending
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
    def src(self, src_: Any) -> None:
        """
        Set source address.

        Parameters
        ----------
        src_: Any
            source address.
        """
        self._src = src_

    def __bytes__(self) -> bytes:
        """Returns message content."""
        return self.in_bytes()

    def __len__(self) -> int:
        """Returns length of the message in bytes."""
        return len(self.in_bytes())

    def __repr__(self) -> str:
        """Returns string representation of the message."""
        return "<{}({}), src={}, dst={}>".format(
            self.__class__.__name__,
            self._content_repr(),
            self._src,
            self._dst
        )

    __str__ = __repr__


class BytesMessage(BaseMessage):

    def __init__(
            self,
            mf_name: str = "std",
            splittable: bool = False,
            slice_length: int = 1024,
            content: bytes = b"",
    ):  # todo: test to default values
        super().__init__(
            mf_name=mf_name,
            splittable=splittable,
            slice_length=slice_length,
        )
        self._content = content

    def set(self, content: bytes | bytearray) -> BytesMessage:
        """
        Set message content.

        Parameters
        ----------
        content: bytes | bytearray
            new message content.

        Returns
        -------
        BytesMessage
            self instance.
        """
        if isinstance(content, bytearray):
            content = bytes(content)
        self._content = content
        return self

    def split(self) -> Generator[BytesMessage, None, None]:
        if not self._splittable:
            yield self
            return

        parts_count = int(np.ceil(len(self) / self._slice_length))
        for start in range(
            0, parts_count * self._slice_length, self._slice_length
        ):
            yield self.get_instance().set(
                self._content[start: start + self._slice_length]
            )

    def in_bytes(self) -> bytes:
        return self._content

    def unpack(self) -> npt.NDArray[np.uint8]:
        return np.frombuffer(self._content, dtype="B")

    def _content_repr(self) -> str:
        return self._content.hex(sep=" ")

    def __add__(self, other: BytesMessage | bytes) -> BytesMessage:
        if isinstance(other, BytesMessage):
            other = bytes(other)
        self._content += other
        return self

    def __getitem__(self, item: int | slice) -> int | bytes:
        return self._content[item]

    def __iter__(self) -> Generator[int, None, None]:
        for byte in self._content:
            yield byte


class MessageFieldsParser(object):
    """
    Represents parser to check the parameters of the fields in the message.

    Parameters
    ----------
    fields: dict[str, FieldType]
        dictionary of fields in the message.
    types: set[type[FieldType]]
        set of field types in the message.
    """

    def __init__(
            self, fields: dict[str, FieldType], types: Set[type[FieldType]]
    ):
        self._fields, self._types = fields, types

    def field_type(self, field_type: type[FieldType]) -> bool:
        """
        Check that message has field of specified type.

        Parameters
        ----------
        field_type: type[FieldType]
            the type of field whose existence is to be checked.

        Returns
        -------
        bool
            True - if field type exists, False - not.
        """
        return field_type in self._types

    @property
    def AddressField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- AddressField exists among fields.
        """
        return AddressField in self._types

    @property
    def CrcField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- CrcField exists among fields.
        """
        return CrcField in self._types

    @property
    def DataField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return DataField in self._types

    @property
    def DataLengthField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return DataLengthField in self._types

    @property
    def OperationField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return OperationField in self._types

    @property
    def ResponseField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return ResponseField in self._types

    @property
    def infinite(self) -> bool:
        """
        Returns
        -------
        bool
            True -- there is a field among the fields that
            has no length limit.
        """
        for field in self._fields.values():
            if not field.finite:
                return True
        return False


class FieldMessage(BaseMessage):
    """
    Represents a message for communication between devices.

    Parameters
    ----------
    mf_name: str, default='std'
        name of the message format.
    splittable: bool, default=False
        shows that the message can be divided by the data.
    slice_length: int, default=1024
        max length of the data in one slice.
    """

    REQUIRED_FIELDS = {"data"}

    def __init__(
            self,
            mf_name: str = "std",
            splittable: bool = False,
            slice_length: int = 1024
    ):  # todo: test to default values
        super().__init__(
            mf_name=mf_name,
            splittable=splittable,
            slice_length=slice_length,
        )

        self._fields: dict[str, FieldType] = {}
        self._field_types: set[type[FieldType]] = set()

    @overload
    def configure(
            self, data: FieldSetter, **fields: FieldSetter
    ) -> FieldMessage:
        ...

    def configure(self, **fields: FieldSetter) -> FieldMessage:
        """
        Configure fields parameters in the message.

        Parameters
        ----------
        **fields : FieldSetter
            fields in format field_name=FieldSetter with field parameters.

        Returns
        -------
        FieldMessage
            object message instance.
        """
        fields_diff = set(self.REQUIRED_FIELDS) - set(fields)
        if len(fields_diff):
            raise ValueError(
                f"not all required fields were got: %s are missing" %
                ", ".join(fields_diff)
            )

        self._fields = {n: self._get_field(n, s) for n, s in fields.items()}

        infinite_exists = False  # todo: create _validate_fields
        for field in self:
            if not field.finite:
                if infinite_exists:
                    raise MessageContentError(
                        self.__class__.__name__,
                        field.name,
                        "second infinite field",
                    )
                infinite_exists = True

        self._field_types = {f.__class__ for f in self}
        self._set_field_ranges()
        return self

    def extract(self, message: bytes) -> FieldMessage:
        """
        Extract fields content from a message.

        May be uses for transform incoming message to a Message class.

        Parameters
        ----------
        message: bytes
            incoming message.

        Returns
        -------
        FieldMessage
            self class instance.

        Raises
        ------
        NotConfiguredMessageError
            if Message has not been configured before.
        """
        if not len(self._fields):
            raise NotConfiguredMessageError(self.__class__.__name__)

        for field in self._fields.values():
            field.extract(message)
        self._validate_content()
        return self

    def get_field_by_type(
            self, field_type: type[FieldType]
    ) -> FieldType | None:
        """
        Get first field by field type.

        Parameters
        ----------
        field_type: type[FieldType]
            the type of field that wanted to find.

        Returns
        -------
        FieldType | None
            field or None if the message has no field of this type.
        """
        for field in self:
            if isinstance(field, field_type):
                return field
        return None

    def get_instance(self) -> FieldMessage:
        return super().get_instance().configure(
            **{f.name: f.get_setter() for f in self}
        )

    @overload
    def set(
            self, data: ContentType = b"", **fields: ContentType
    ) -> FieldMessage:  # todo: check that data is DataField
        ...

    def set(self, **fields: ContentType) -> FieldMessage:
        """
        Set fields content by names.

        Parameters
        ----------
        fields: Content
            field content in format field_name=content.

        Returns
        -------
        FieldMessage
            self class instance.

        Raises
        ------
        NotConfiguredMessageError
            if Message has not been configured before.
        """
        if not len(self._fields):
            raise NotConfiguredMessageError(self.__class__.__name__)

        for name, content in fields.items():
            self[name].set(content)

        if self.has.CrcField:
            crc = self.get_field_by_type(CrcField)
            if crc.name not in fields:
                crc.update()

        self._validate_content()
        return self

    # todo: parts count for bytes but split by data with int count of word
    def split(self) -> Generator[FieldMessage, None, None]:
        """
        Split data field on slices.

        Yields
        ------
        Message
            message part.
        """
        if not self._splittable:
            yield self
            return

        if self.has.AddressField \
                and self.has.OperationField \
                and self.has.DataLengthField:
            address = self.get_field_by_type(AddressField)
            operation = self.get_field_by_type(OperationField)
            data_length = self.get_field_by_type(DataLengthField)
            parts_count = int(np.ceil(
                data_length[0] / self._slice_length
            ))

            for i_part in range(parts_count):
                if i_part == parts_count - 1:
                    data_len = data_length[0] - i_part * self._slice_length
                else:
                    data_len = self._slice_length

                msg = deepcopy(self)

                if operation.base == 'r':
                    msg.get_field_by_type(DataLengthField).set(data_len)
                elif operation.base == 'w':
                    start = i_part * self._slice_length
                    end = start + data_len
                    if data_length.units == DataLengthField.WORDS:
                        msg.data.set(self.data[start:end])
                    elif data_length.units == DataLengthField.BYTES:
                        msg.data.set(self.data.content[start:end])
                    else:
                        raise TypeError('Unsupported data units')
                    msg.get_field_by_type(DataLengthField).update()
                msg.get_field_by_type(AddressField).set(
                    address[0] + i_part * self._slice_length)
                msg.set_src_dst(self._src, self._dst)
                yield msg

    def in_bytes(self) -> bytes:
        return b"".join(map(bytes, self._fields.values()))

    def unpack(self) -> npt.NDArray:
        unpacked = np.array([])  # todo: create array with required length
        for field in self:
            unpacked = np.append(unpacked, field.unpack())
        return unpacked

    def _content_repr(self) -> str:
        """
        Convert content to string

        Returns
        -------

        """
        return ", ".join(map(
            lambda x: "%s=%s" % x, self._fields.items()
        ))

    def _get_field(self, name: str, setter: FieldSetter) -> FieldType:
        """
        Returns an instance of a field with specified parameters from
        the setter.

        Parameters
        ----------
        name : str
            field name.
        setter : FieldSetter
            setter with field parameters.

        Returns
        -------
        FieldType
            field instance.
        """
        return setter.get_field_class()(
            self._mf_name,
            name,
            start_byte=0,
            parent=self,
            **setter.kwargs
        )

    def _set_field_ranges(self) -> None:
        """Set fields start and stop bytes."""
        fields: list[FieldType] = list(self)
        i_infinite, next_start_byte = -1, 0
        for i_field, field in enumerate(fields):

            field.start_byte = next_start_byte
            next_start_byte += field.expected * field.bytesize

            if field.finite:
                field.stop_byte = next_start_byte
            elif i_field == len(fields) - 1:
                field.stop_byte = None
                return
            else:
                i_infinite = i_field
                break

        if i_infinite < 0:
            return

        next_start_byte = 0
        for i_field, field in enumerate(fields[::-1]):

            if field.finite:
                if next_start_byte:
                    field.stop_byte = next_start_byte
                else:
                    field.stop_byte = None
                next_start_byte -= field.expected * field.bytesize
                field.start_byte = next_start_byte

            else:
                if i_infinite:  # i_infinite > 0
                    field.start_byte = fields[i_infinite - 1].stop_byte
                    field.stop_byte = fields[i_infinite + 1].start_byte
                else:  # i_infinite == 0
                    field.stop_byte = fields[1].start_byte
                break

    def _validate_content(self) -> None:
        """Validate content."""
        for field in self:
            if not (field.words_count or field.may_be_empty):
                raise MessageContentError(
                    self.__class__.__name__, field.name, "field is empty"
                )

        if self.has.OperationField and self.has.DataLengthField:
            operation = self.get_field_by_type(OperationField)
            data_length = self.get_field_by_type(DataLengthField)

            if operation.base == "w" and (
                    not self.data.may_be_empty or self.data.words_count
            ) and data_length[0] != data_length.calculate(self.data):
                raise MessageContentError(
                    self.__class__.__name__,
                    data_length.name,
                    "invalid length"
                )

        if self.has.CrcField:
            crc = self.get_field_by_type(CrcField)
            res_crc, ref_crc = crc.calculate(self), crc[0]
            if ref_crc != res_crc:
                raise MessageContentError(
                    self.__class__.__name__,
                    crc.name,
                    "invalid crc value, '%x' != '%x'" % (ref_crc, res_crc)
                )

    @property
    def data(self) -> DataField:
        """
        Returns
        -------
        DataField
            data field instance.
        """
        return self._fields["data"]

    @property
    def has(self) -> MessageFieldsParser:
        """
        Returns
        -------
        MessageFieldsParser
            parser to check the parameters of the fields in the message.
        """
        return MessageFieldsParser(self._fields, self._field_types)

    @property
    def response_codes(self) -> dict[str, Code]:
        """
        Returns
        -------
        dict of {str, Code}
            dictionary of all response codes in the message, where
            key is the field name and value is the code.
            If there is no ResponseField in the message,
            the dictionary will be empty.
        """
        codes = {}
        for field in self._fields.values():
            if isinstance(field, ResponseField):
                codes[field.name] = field.current_code
        return codes

    def __add__(self, other: FieldMessage | bytes) -> FieldMessage:
        """
        Add new data to the data field.

        If the passed argument is the bytes type, the passed argument
        is added to the data field (right concatenation).

        If the passed argument is an instance of the same class or is
        a child, then a content and class validation.

        In both cases it is checked before writing that the new data
        will contain an integer number of words in the format of the current.
        instance

        Parameters
        ----------
        other: bytes or FieldMessage
            additional data or message.

        Returns
        -------
        FieldMessage
            self instance.
        """

        if isinstance(other, FieldMessage):
            if other.mf_name != self._mf_name:
                raise TypeError(
                    "messages have different formats: %s != %s" % (
                        other.mf_name, self._mf_name
                    )
                )
            other = other.data.content
        elif isinstance(other, bytes):
            pass
        else:
            raise TypeError(
                "%s cannot be added to the message" % type(other)
            )

        self.data.append(other)
        if self.has.DataLengthField:
            self.get_field_by_type(DataLengthField).update()
        return self

    def __getitem__(self, item: str) -> FieldType:
        """
        Returns a field instance by field name.

        Parameters
        ----------
        item : str
            field name.

        Returns
        -------
        FieldType
            field instance.
        """
        return self._fields[item]

    def __iter__(self) -> Generator[FieldType, None, None]:
        """
        Iteration by fields.

        Yields
        -------
        FieldType
            field instance.
        """
        for field in self._fields.values():
            yield field


class StrongFieldMessage(FieldMessage):

    REQUIRED_FIELDS = FieldMessage.REQUIRED_FIELDS.union(
        {"address", "data_length", "operation"}
    )

    @overload
    def configure(
            self,
            *,
            address: FieldSetter,
            data: FieldSetter,
            data_length: FieldSetter,
            operation: FieldSetter,
            **fields: FieldSetter,
    ) -> StrongFieldMessage:
        ...

    def configure(self, **fields: FieldSetter) -> StrongFieldMessage:
        super().configure(**fields)
        return self

    @overload
    def set(
            self,
            *,
            address: ContentType = b"",
            operation: ContentType = b"",
            data: ContentType = b"",
            data_length: ContentType = b"",
            **fields: ContentType
    ) -> StrongFieldMessage:
        ...

    def set(self, **fields: ContentType) -> StrongFieldMessage:
        if not len(self._fields):
            raise NotConfiguredMessageError(self.__class__.__name__)

        for name, content in fields.items():
            self[name].set(content)
        if "data_length" not in fields:
            self.data_length.update()
        if "crc" in self._fields and "crc" not in fields \
                and isinstance(self["crc"], CrcField):
            self["crc"].update()
        self._validate_content()
        return self

    # todo: parts count for bytes but split by data with int count of word
    def split(self) -> Generator[FieldMessage, None, None]:
        """
        Split data field on slices.

        Yields
        ------
        Message
            message part.
        """
        if not self._splittable:
            yield self
            return

        parts_count = int(np.ceil(
            self.data_length[0] / self._slice_length
        ))
        for i_part in range(parts_count):
            if i_part == parts_count - 1:
                data_len = self.data_length[0] - i_part * self._slice_length
            else:
                data_len = self._slice_length

            msg = deepcopy(self)

            if msg.operation.base == 'r':
                msg.data_length.set(data_len)
            elif msg.operation.base == 'w':
                start = i_part * self._slice_length
                end = start + data_len
                if self.data_length.units == DataLengthField.WORDS:
                    msg.data.set(self.data[start:end])
                elif self.data_length.units == DataLengthField.BYTES:
                    msg.data.set(self.data.content[start:end])
                else:
                    raise TypeError('Unsupported data units')
                msg.data_length.update()
            msg.address.set(
                self.address[0] + i_part * self._slice_length)
            msg.set_src_dst(self._src, self._dst)
            yield msg

    def _validate_content(self) -> None:
        """Validate content."""
        for field in self:
            if not (field.words_count or field.may_be_empty):
                raise MessageContentError(
                    self.__class__.__name__, field.name, "field is empty"
                )

        if self.operation.base == "w" and (
                not self.data.may_be_empty or self.data.words_count
        ) and self.data_length[0] != self.data_length.calculate(self.data):
            raise MessageContentError(
                self.__class__.__name__,
                self.data_length.name,
                "invalid length"
            )

        if "crc" in self._fields and isinstance(self["crc"], CrcField):
            crc: CrcField = self["crc"]
            res_crc, ref_crc = crc.calculate(self), crc.unpack()[0]
            if ref_crc != res_crc:
                raise MessageContentError(
                    self.__class__.__name__,
                    crc.name,
                    "invalid crc value, '%x' != '%x'" % (ref_crc, res_crc)
                )

    @property
    def address(self) -> AddressField:
        """
        Returns
        -------
        AddressField
            address field instance.
        """
        return self._fields["address"]

    @property
    def data_length(self) -> DataLengthField:
        """
        Returns
        -------
        DataLengthField
            data length field instance.
        """
        return self._fields["data_length"]

    @property
    def operation(self) -> OperationField:
        """
        Returns
        -------
        OperationField
            operation field instance.
        """
        return self._fields["operation"]

    def __add__(self, other: FieldMessage | bytes) -> FieldMessage:
        super().__add__(other)
        self.data_length.update()
        return self


MessageType = (
    BytesMessage
    | FieldMessage
    | StrongFieldMessage
)


@dataclass
class MessageSetter(object):
    """
    Represent setter, which contain keyword arguments for setting message.
    """

    message_type: str = "bytes"
    "type of a message class."

    mf_name: str = "std"
    "name of the message format."

    splittable: bool = False
    "shows that the message can be divided by the data."

    slice_length: int = 1024
    "max length of the data in one slice."

    MESSAGE_TYPES: ClassVar[dict[str, type[BaseMessage | MessageType]]] = {
        "base": BaseMessage,
        "bytes": BytesMessage,
        "field": FieldMessage,
        "strong_field": StrongFieldMessage,
    }

    def __post_init__(self):
        if self.message_type == "base":
            raise ValueError("BaseMessage not supported by setter")
        if self.message_type not in self.MESSAGE_TYPES:
            raise ValueError("invalid message type: %r" % self.message_type)

    @property
    def kwargs(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict[str, Any]
            keywords arguments for setting a message.
        """
        return dict(
            mf_name=self.mf_name,
            splittable=self.splittable,
            slice_length=self.slice_length,
        )

    @property
    def message(self) -> MessageType:
        """
        Returns
        -------
        MessageType
            initialized message.
        """
        return self.message_class(**self.kwargs)

    @property
    def message_class(self) -> type[MessageType]:
        """
        Returns
        -------
        type[MessageType]
            message class by message type.
        """
        return self.MESSAGE_TYPES[self.message_type]

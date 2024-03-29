from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Generator,
    ClassVar,
    Any,
    overload,
)

import numpy as np
import numpy.typing as npt

from .field import (
    ContentType,
    AddressField,
    CrcField,
    DataField,
    DataLengthField,
    IdField,
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
    "FieldMessage",
    "SingleFieldMessage",
    "StrongFieldMessage",
    "MessageSetter",
    "MessageContentError",
    "NotConfiguredMessageError",
]

# todo: add tests


class MessageFieldsGetParser(object):
    """
    Represents parser to get the field from message.

    Parameters
    ----------
    fields: dict[str, FieldType]
        dictionary of fields in the message.
    types: dict[type[FieldType], str]
        set of field types in the message.
    """

    def __init__(
            self,
            fields: dict[str, FieldType],
            types: dict[type[FieldType], str],
    ):
        self._fields, self._types = fields, types

    def field_by_type(self, type_: type[FieldType]) -> FieldType:
        """
        Get first field with specified type.

        Parameters
        ----------
        type_: type[FieldType]

        Returns
        -------
        FieldType
            field if specified type.

        Raises
        ------
        TypeError
            if type not found in fields list.
        """
        if type_ not in self._types:
            raise TypeError("there is no field with type %s" % type_.__name__)
        return self[self._types[type_]]

    @property
    def AddressField(self) -> AddressField:
        """
        Returns
        -------
        AddressField
            address field instance.
        """
        return self.field_by_type(AddressField)

    @property
    def CrcField(self) -> CrcField:
        """
        Returns
        -------
        CrcField
            crc field instance.
        """
        return self.field_by_type(CrcField)

    @property
    def DataField(self) -> DataField:
        """
        Returns
        -------
        DataField
            data field instance.
        """
        return self.field_by_type(DataField)

    @property
    def DataLengthField(self) -> DataLengthField:
        """
        Returns
        -------
        DataLengthField
            data length field instance.
        """
        return self.field_by_type(DataLengthField)

    @property
    def IdField(self) -> IdField:
        """
        Returns
        -------
        IdField
            id field instance.
        """
        return self.field_by_type(IdField)

    @property
    def OperationField(self) -> OperationField:
        """
        Returns
        -------
        OperationField
            operation length field instance.
        """
        return self.field_by_type(OperationField)

    @property
    def ResponseField(self) -> ResponseField:
        """
        Returns
        -------
        ResponseField
            response length field instance.
        """
        return self.field_by_type(ResponseField)

    def __getitem__(self, name: str) -> FieldType:
        """
        Get field by unique name.

        Parameters
        ----------
        name: str
            field name.

        Returns
        -------
        FieldType
            field instance.
        """
        if name not in self._fields:
            raise ValueError("there is no field with name %s" % name)
        return self._fields[name]


class MessageFieldsHasParser(object):
    """
    Represents parser to check the parameters of the fields in the message.

    Parameters
    ----------
    fields: dict[str, FieldType]
        dictionary of fields in the message.
    types: dict[type[FieldType], str]
        set of field types in the message.
    """

    def __init__(
            self,
            fields: dict[str, FieldType],
            types: dict[type[FieldType], str],
    ):
        self._fields, self._types = fields, types

    def field_type(self, type_: type[FieldType]) -> bool:
        """
        Check that message has field of specified type.

        Parameters
        ----------
        type_: type[FieldType]
            the type of field whose existence is to be checked.

        Returns
        -------
        bool
            True - if field type exists, False - not.
        """
        return type_ in self._types

    @property
    def AddressField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- AddressField exists among fields.
        """
        return self.field_type(AddressField)

    @property
    def CrcField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- CrcField exists among fields.
        """
        return self.field_type(CrcField)

    @property
    def DataField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return self.field_type(DataField)

    @property
    def DataLengthField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return self.field_type(DataLengthField)

    @property
    def IdField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- IdField exists among fields.
        """
        return self.field_type(IdField)

    @property
    def OperationField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return self.field_type(OperationField)

    @property
    def ResponseField(self) -> bool:
        """
        Returns
        -------
        bool
            True -- DataField exists among fields.
        """
        return self.field_type(ResponseField)

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


class FieldMessage(object):
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
        super().__init__()
        self._mf_name = mf_name
        self._splittable = splittable
        self._slice_length = slice_length

        self._src, self._dst = None, None
        self._fields: dict[str, FieldType] = {}
        self._field_types: dict[type[FieldType], str] = {}

    def clear_src_dst(self) -> None:
        """Set `src` and `dst` to None."""
        self._dst, self._src = None, None

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

        self._field_types.clear()
        for field in self:
            if field.__class__ not in self._field_types:
                self._field_types[field.__class__] = field.name
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

    def get_instance(self) -> FieldMessage:
        """
        Get the same class as the current object, initialized with
        the same arguments, but with empty content.

        Returns
        -------
        FieldMessage
            new class instance.
        """
        return self.__class__(
            mf_name=self._mf_name,
            splittable=self._splittable,
            slice_length=self._slice_length,
        ).configure(
            **{f.name: f.get_setter() for f in self}
        )

    def get_setter(self) -> MessageSetter:
        """
        Get setter of the message instance.

        Returns
        -------
        MessageSetter
            setter for this message instance.
        """
        message_type = {
            v: k for k, v in MessageSetter.MESSAGE_TYPES.items()
        }[self.__class__]
        return MessageSetter(
            message_type=message_type,
            mf_name=self._mf_name,
            splittable=self._splittable,
            slice_length=self.slice_length,
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

        if self.has.DataLengthField:
            data_length = self.get.DataLengthField
            if data_length.name not in fields:
                data_length.update()

        if self.has.CrcField:
            crc = self.get.CrcField
            if crc.name not in fields:
                crc.update()

        self._validate_content()
        return self

    def set_src_dst(self, src: Any, dst: Any) -> FieldMessage:
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

    # todo: parts count for bytes but split by data with int count of word
    def split(self) -> Generator[FieldMessage, None, None]:
        """
        Split data field on slices.

        Yields
        ------
        FieldMessage
            message part.
        """
        if not self._splittable:
            yield self
            return

        if self.has.AddressField \
                and self.has.OperationField \
                and self.has.DataLengthField:  # todo: refactor
            address = self.get.AddressField
            operation = self.get.OperationField
            data_length = self.get.DataLengthField
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
                    msg.get.DataLengthField.set(data_len)
                elif operation.base == 'w':
                    start = i_part * self._slice_length
                    end = start + data_len
                    if data_length.units == Code.WORDS:
                        msg.data.set(self.data[start:end])
                    elif data_length.units == Code.BYTES:
                        msg.data.set(self.data.content[start:end])
                    else:
                        raise TypeError('Unsupported data units')
                    msg.get.DataLengthField.update()
                msg.get.AddressField.set(
                    address[0] + i_part * self._slice_length)
                msg.set_src_dst(self._src, self._dst)
                yield msg

    def in_bytes(self) -> bytes:
        return b"".join(map(bytes, self._fields.values()))

    def unpack(self) -> npt.NDArray:
        """
        Returns
        -------
        numpy.typing.NDArray
            unpacked message content.
        """
        unpacked = np.array([])  # todo: create array with required length
        for field in self:
            unpacked = np.append(unpacked, field.unpack())
        return unpacked

    def _content_repr(self) -> str:
        """
        Convert content to string

        Returns
        -------
        str
            string representation of the content.
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
            operation = self.get.OperationField
            data_length = self.get.DataLengthField

            if operation.base == "w" and (
                    not self.data.may_be_empty or self.data.words_count
            ) and data_length[0] != data_length.calculate(self.data):
                raise MessageContentError(
                    self.__class__.__name__,
                    data_length.name,
                    "invalid length"
                )

        if self.has.CrcField:
            crc = self.get.CrcField
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
        return self.get.DataField

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
    def get(self) -> MessageFieldsGetParser:
        """
        Returns
        -------
        MessageFieldsGetParser
            parser for getting fields instance.
        """
        return MessageFieldsGetParser(self._fields, self._field_types)

    @property
    def has(self) -> MessageFieldsHasParser:
        """
        Returns
        -------
        MessageFieldsHasParser
            parser to check the parameters of the fields in the message.
        """
        return MessageFieldsHasParser(self._fields, self._field_types)

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
        if ResponseField not in self._field_types:
            return {}

        codes = {}
        for field in self._fields.values():
            if isinstance(field, ResponseField):
                codes[field.name] = field.current_code
        return codes

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
        pyiak_instr.communication.Message.split: method for splitting
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
            self.get.DataLengthField.update()
        return self

    def __bytes__(self) -> bytes:
        """Returns message content."""
        return self.in_bytes()

    def __getitem__(self, name: str) -> FieldType:
        """
        Returns a field instance by field name.

        Parameters
        ----------
        name : str
            field name.

        Returns
        -------
        FieldType
            field instance.
        """
        return self.get[name]

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


class SingleFieldMessage(FieldMessage):
    """
    Represents class for message with single field.

    Parameters
    ----------
    data: ContentType, default=b""
        content of the message.
    data__fmt: str, default="B"
        data format of the content.
    mf_name: str, default='std'
        name of the message format.
    splittable: bool, default=False
        shows that the message can be divided by the data.
    slice_length: int, default=1024
        max length of the data in one slice.
    """

    def __init__(
            self,
            data: ContentType = b"",
            data__fmt: str = "B",
            mf_name: str = "std",
            splittable: bool = False,
            slice_length: int = 1024,
    ):
        super().__init__(
            mf_name=mf_name,
            slice_length=slice_length,
            splittable=splittable,
        )
        self.configure(data=FieldSetter.data(expected=-1, fmt=data__fmt))
        self.set(data=data)

    def configure(self, data: FieldSetter) -> SingleFieldMessage:
        if data.kwargs["expected"] >= 0:
            raise ValueError("data field must be infinite")
        super().configure(data=data)
        return self

    def set(self, data: ContentType) -> SingleFieldMessage:
        super().set(data=data)
        return self


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
        super().set(**fields)
        return self

    @property
    def address(self) -> AddressField:
        """
        Returns
        -------
        AddressField
            address field instance.
        """
        return self.get.AddressField

    @property
    def data_length(self) -> DataLengthField:
        """
        Returns
        -------
        DataLengthField
            data length field instance.
        """
        return self.get.DataLengthField

    @property
    def operation(self) -> OperationField:
        """
        Returns
        -------
        OperationField
            operation field instance.
        """
        return self.get.OperationField


MessageType = (
    FieldMessage
    | SingleFieldMessage
    | StrongFieldMessage
)


@dataclass
class MessageSetter(object):
    """
    Represent setter, which contain keyword arguments for setting message.
    """

    message_type: str = "field"
    "type of a message class."

    mf_name: str = "std"
    "name of the message format."

    splittable: bool = False
    "shows that the message can be divided by the data."

    slice_length: int = 1024
    "max length of the data in one slice."

    MESSAGE_TYPES: ClassVar[dict[str, type[MessageType]]] = {
        "field": FieldMessage,
        "single": SingleFieldMessage,
        "strong": StrongFieldMessage,
    }

    def __post_init__(self):
        if self.message_type == "base":
            raise ValueError("BaseMessage not supported by setter")
        if self.message_type not in self.MESSAGE_TYPES:
            raise ValueError("invalid message type: %r" % self.message_type)

    @property
    def init_kwargs(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict[str, Any]
            keywords arguments for setting a MessageSetter.
        """
        return dict(
            message_type=self.message_type,
            **self.kwargs
        )

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

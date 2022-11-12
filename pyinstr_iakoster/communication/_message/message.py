from __future__ import annotations
from copy import deepcopy
from typing import Generator, Any, overload

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
from pyinstr_iakoster.exceptions import (
    MessageContentError,
    NotConfiguredMessageError,
)
from pyinstr_iakoster.core import Code


__all__ = [
    "BytesMessage",
    "Message",
    "MessageContentError",
    "NotConfiguredMessageError",
]

# todo: add tests


class BaseMessage(object):

    def __init__(
            self,
            mf_name: str = "std",
            splitable: bool = False,
            slice_length: int = 1024,
    ):
        self._mf_name = mf_name
        self._splitable = splitable
        self._slice_length = slice_length

        self._src, self._dst = None, None

    def hex(self, sep: str = " ", sep_step: int = None) -> str:
        """
        Returns a string of hexadecimal numbers from the content.

        Parameters
        ----------
        sep: str
            separator between bytes/words and fields.
        sep_step: int
            separator step.

        Returns
        -------
        str
            hex string.
        """
        raise NotImplementedError()

    def set(self, *args, **kwargs) -> BaseMessage:
        """
        Set message content.

        Returns
        -------
        BytesMessage
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

    def to_bytes(self) -> bytes:
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

    def __iter__(self):
        """Iteration by message."""
        raise NotImplementedError()

    def clear_src_dst(self) -> None:
        """Set src and dst to None."""
        self._dst, self._src = None, None

    def get_instance(self, **kwargs: Any) -> BaseMessage:
        """
        Get the same class as the current object, initialized with
        the specified arguments.

        Parameters
        ----------
        **kwargs: Any
            initial keywords arguments.

        Returns
        -------
        Message
            new class instance.
        """
        return self.__class__(**kwargs)

    def get_same_instance(self) -> BaseMessage:
        """
        Get the same class as the current object, initialized with
        the same arguments, but with empty content.

        Returns
        -------
        Message
            new class instance.
        """
        return self.__class__(
            mf_name=self._mf_name,
            splitable=self._splitable,
            slice_length=self._slice_length,
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
        Message
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
    def splitable(self) -> bool:
        """
        Indicates that the message can be splited.

        Returns
        -------
        bool
            pointer to the possibility of separation
        """
        return self._splitable

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
        return self.to_bytes()

    def __len__(self) -> int:
        """Returns length of the message in bytes."""
        return len(self.to_bytes())

    def __repr__(self) -> str:
        """Returns string representation of the message."""
        return "<{}({}), src={}, dst={}>".format(
            self.__class__.__name__,
            self._content_repr(),
            self._src,
            self._dst
        )

    def __str__(self) -> str:
        """Returns fields converted to string."""
        return self.hex().upper()


class BytesMessage(BaseMessage):

    def __init__(
            self,
            mf_name: str = "std",
            content: bytes = b"",
            splitable: bool = False,
            slice_length: int = 1024,
    ):
        super().__init__(
            mf_name=mf_name,
            splitable=splitable,
            slice_length=slice_length,
        )
        self._content = content

    def hex(self, sep: str = " ", sep_step: int = None) -> str:
        return self._content.hex(sep=sep, bytes_per_sep=sep_step)

    def set(self, content: bytes) -> BytesMessage:
        """
        Set message content.

        Parameters
        ----------
        content: bytes
            new message content.

        Returns
        -------
        BytesMessage
            self instance.
        """
        self._content = content
        return self

    def to_bytes(self) -> bytes:
        return self._content

    def unpack(self) -> npt.NDArray:
        return np.frombuffer(self._content, dtype="B")

    def _content_repr(self) -> str:
        return self.hex()

    def __add__(self, other: BytesMessage | bytes) -> BytesMessage:
        self._content += bytes(other)
        return self

    def __getitem__(self, item: int | slice) -> int | bytes:
        return self._content[item]

    def __iter__(self) -> Generator[int, None, None]:
        for byte in self._content:
            yield byte


class Message(BaseMessage):
    """
    Represents a message for communication between devices.

    Parameters
    ----------
    mf_name: str
        name of the message format.
    splitable: bool
        shows that the message can be divided by the data.
    slice_length: int
        max length of the data in one slice.
    """

    REQ_FIELDS = {
        "address": AddressField,
        "data": DataField,
        "data_length": DataLengthField,
        "operation": OperationField,
    }
    SPECIAL_FIELDS = {
        "crc": CrcField,
        "single": SingleField,
        "static": StaticField,
        "response": ResponseField,
    }

    def __init__(
            self,
            mf_name: str = "std",
            splitable: bool = False,
            slice_length: int = 1024
    ):
        super().__init__(
            mf_name=mf_name,
            splitable=splitable,
            slice_length=slice_length,
        )

        self._fields: dict[str, FieldType] = {}

    @overload
    def configure(
            self,
            *,
            address: FieldSetter,
            operation: FieldSetter,
            data: FieldSetter = FieldSetter(),
            data_length: FieldSetter = FieldSetter(),
            **fields: FieldSetter
    ) -> Message:
        ...

    def configure(self, **fields: FieldSetter) -> Message:
        """
        Configure fields parameters in the message.

        Parameters
        ----------
        **fields : FieldSetter
            fields in format field_name=FieldSetter with field parameters.

        Returns
        -------
        Message
            object message instance.
        """
        fields_diff = set(self.REQ_FIELDS) - set(fields)
        if len(fields_diff):
            raise ValueError(
                f"not all requared fields were got: %s are missing" %
                ", ".join(fields_diff)
            )
        del fields_diff

        def get_setters() -> FieldType:
            last_name = ""
            for setter_name, setter_ in fields.items():
                if infinite is not None:
                    break
                yield setter_name, setter_
                last_name = setter_name

            if len(last_name) and infinite is not None:
                for setter_name, setter_ in list(fields.items())[::-1]:
                    if setter_name == last_name:
                        break
                    yield setter_name, setter_

        self._fields.clear()
        next_start_byte, footers, infinite = 0, {}, None
        selected = self._fields

        for name, setter in get_setters():
            field = self._get_field(name, next_start_byte, setter)
            selected[name] = field
            if field.finite:
                offset = field.expected * field.bytesize
                if infinite is not None:
                    offset *= -1
                next_start_byte += offset
                if infinite is not None:
                    field.start_byte = next_start_byte
                    field.stop_byte += offset

            elif infinite is not None:
                raise MessageContentError(
                    self.__class__.__name__, name, "second infinite field"
                )
            else:
                next_start_byte, selected, infinite = 0, footers, field

        if infinite is not None and len(footers):
            footers = list(footers.items())[::-1]
            infinite.stop_byte = footers[0][1].start_byte
            footers[-1][1].stop_byte = None
            for i_field, (name, field) in enumerate(footers):
                self._fields[name] = field

        return self

    def extract(self, message: bytes) -> Message:
        """
        Extract fields content from a message.

        May be uses for transform incoming message to a Message class.

        Parameters
        ----------
        message: bytes
            incoming message.

        Returns
        -------
        Message
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

    def get_same_instance(self) -> Message:
        return super().get_same_instance().configure(
            **{n: f.get_setter() for n, f in self._fields.items()}
        )

    def hex(self, sep: str = " ", sep_step: int = None) -> str:
        fields_hex = []
        for field in self:
            field_hex = field.hex(sep=sep, sep_step=sep_step)
            if field_hex != "":
                fields_hex.append(field_hex)
        return sep.join(fields_hex)

    @overload
    def set(
            self,
            *,
            address: ContentType,
            operation: ContentType,
            data: ContentType = b"",
            data_length: ContentType = b"",
            **fields: ContentType
    ) -> Message:
        ...

    def set(self, **fields: ContentType) -> Message:
        """
        Set field content by names.

        Parameters
        ----------
        fields: Content
            field content in format field_name=content.

        Returns
        -------
        Message
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
        if "data_length" not in fields:
            self.data_length.update()
        if "crc" in self._fields and "crc" not in fields \
                and isinstance(self["crc"], CrcField):
            self["crc"].update()
        self._validate_content()
        return self

    def split(self) -> Generator[Message, None, None]:  # todo: yield full message if not splitable
        """
        Split data field on slices.

        Yields
        ------
        Message
            message part.

        Raises
        ------
        TypeError
            if message is not cutable by data.
        """
        if not self._splitable:
            raise TypeError(
                f"{self.__class__.__name__} cannot be cut into parts"
            )

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
                    raise TypeError('Unsuppoted data units')
                msg.data_length.update()
            msg.address.set(
                self.address[0] + i_part * self._slice_length)
            msg.set_src_dst(self._src, self._dst)
            yield msg

    def to_bytes(self) -> bytes:
        return b"".join(
            bytes(field) for field in self._fields.values()
        )

    def unpack(self) -> npt.NDArray:
        unpacked = np.array([])
        for field in self:
            unpacked = np.append(unpacked, field.unpack())
        return unpacked

    def _content_repr(self) -> str:
        gen = zip(self._fields, self)
        content_repr = "%s=%s" % next(gen)
        for name, field in gen:
            content_repr += ", %s=" % name
            if field.words_count:
                content_repr += str(field)
            else:
                content_repr += "EMPTY"
        return content_repr

    def _get_field(
            self, name: str, start_byte: int, setter: FieldSetter
    ) -> FieldType:
        """
        Returns an instance of a field with specified parameters from
        the setter.

        Parameters
        ----------
        name : str
            field name.
        start_byte : int
            start byte of the field in a message.
        setter : FieldSetter
            setter with field parameters.

        Returns
        -------
        FieldType
            field instance.
        """
        if name in self.REQ_FIELDS:
            return self.REQ_FIELDS[name](
                self._mf_name,
                start_byte=start_byte,
                parent=self,
                **setter.kwargs
            )
        return self.SPECIAL_FIELDS.get(setter.special, Field)(
            self._mf_name,
            name,
            start_byte=start_byte,
            parent=self,
            **setter.kwargs
        )

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
    def data(self) -> DataField:
        """
        Returns
        -------
        DataField
            data field instance.
        """
        return self._fields["data"]

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
    def have_infinite(self) -> bool:
        """
        Returns
        -------
        bool
            mark that there is a field in the message that can have
            an unlimited length.
        """
        for field in self._fields.values():
            if not field.finite:
                return True
        return False

    @property
    def operation(self) -> OperationField:
        """
        Returns
        -------
        OperationField
            operation field instance.
        """
        return self._fields["operation"]

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

    def __add__(self, other: Message | bytes) -> Message:
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
        other: bytes or Message
            additional data or message.

        Returns
        -------
        Message
            self instance.
        """

        match other:
            case Message():
                if other.mf_name != self._mf_name:
                    raise TypeError(
                        "messages have different formats: %s != %s" % (
                            other.mf_name, self._mf_name
                        )
                    )
                other = other.data.content
            case bytes():
                pass
            case _:
                raise TypeError(
                    "%s cannot be added to the message" % type(other)
                )

        self.data.append(other)
        self.data_length.update()
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

    # todo: check with other str methods (hex, __str__, __repr__)
    #  in Field and Message
    def __str__(self) -> str:
        return " ".join(str(field) for field in self if len(str(field)))

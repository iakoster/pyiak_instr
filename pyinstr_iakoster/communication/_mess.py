from copy import deepcopy
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from ._fields import (
    Content,
    Fields,
    Field,
    FieldSingle,
    FieldStatic,
    FieldAddress,
    FieldData,
    FieldDataLength,
    FieldOperation,
)
from ..exceptions import (
    MessageContentError,
    NotConfiguredMessageError,
    FloatWordsCountError,
    PartialFieldError,
)


__all__ = [
    "Message",
    "FieldSetter",
    "Field",
    "FieldSingle",
    "FieldStatic",
    "FieldAddress",
    "FieldData",
    "FieldDataLength",
    "FieldOperation",
    "Content",
    "MessageContentError",
    "NotConfiguredMessageError",
    "FloatWordsCountError",
    "PartialFieldError",
]


class FieldSetter(object):

    BYTES = FieldDataLength.BYTES
    WORDS = FieldDataLength.WORDS

    def __init__(
            self,
            special: str = None,
            **kwargs: Any,
    ):
        self.special = special
        self.kwargs = kwargs

    @classmethod
    def base(
            cls,
            *,
            expected: int,
            fmt: str,
            info: dict[str, Any] = None,
            may_be_empty: bool = False,
    ):
        """For classical field"""
        return cls(
            expected=expected,
            fmt=fmt,
            info=info,
            may_be_empty=may_be_empty
        )

    @classmethod
    def single(
            cls,
            *,
            fmt: str,
            info: dict[str, Any] = None,
            may_be_empty: bool = False,
    ):
        return cls(
            special="single",
            fmt=fmt,
            info=info,
            may_be_empty=may_be_empty
        )

    @classmethod
    def static(
            cls,
            *,
            fmt: str,
            content: Content,
            info: dict[str, Any] = None,
    ):
        return cls(
            special="static",
            fmt=fmt,
            content=content,
            info=info,
        )

    @classmethod
    def address(
            cls,
            *,
            fmt: str,
            info: dict[str, Any] | None = None
    ):
        return cls(fmt=fmt, info=info)

    @classmethod
    def data(
            cls,
            *,
            expected: int,
            fmt: str,
            info: dict[str, Any] | None = None
    ):
        return cls(expected=expected, fmt=fmt, info=info)

    @classmethod
    def data_length(
            cls,
            *,
            fmt: str,
            units: int = BYTES,
            additive: int = 0,
            info: dict[str, Any] | None = None
    ):
        return cls(fmt=fmt, units=units, additive=additive, info=info)

    @classmethod
    def operation(
            cls,
            *,
            fmt: str,
            desc_dict: dict[str, int] = None,
            info: dict[str, Any] | None = None
    ):
        return cls(fmt=fmt, desc_dict=desc_dict, info=info)


class MessageBase(object):
    """
    Base class of the Message with required methods.
    """

    REQ_FIELDS = {
        "address": FieldAddress,
        "data": FieldData,
        "data_length": FieldDataLength,
        "operation": FieldOperation,
    }
    SPECIAL_FIELDS = {
        "single": FieldSingle,
        "static": FieldStatic,
    }

    # _addr: FieldAddress
    # _data: FieldData
    # _dlen: FieldDataLength
    # _oper: FieldOperation

    def __init__(
            self,
            format_name: str = "default",
            splitable: bool = False,
            slice_length: int = 1024
    ):
        self._fmt_name = format_name
        self._splitable = splitable
        self._slice_length = slice_length
        self._fields: dict[str, Fields] = {}
        self._configured = False
        self._tx, self._rx = None, None

        self._kwargs = dict(
            format_name=format_name,
            splitable=splitable,
            slice_length=slice_length,
        )
        self._configured_fields: dict[str, FieldSetter] = {}

    def clear_addresses(self) -> None:
        """Set addresses to None."""
        self._tx, self._rx = None, None

    def get_instance(self, **kwargs: Any):
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

    def set_addresses(self, tx: Any = None, rx: Any = None):
        """
        Set Tx and Rx addresses.

        Addresses may differ depending on the type of connection used.
            * Tx - is a source address;
            * Rx - is a reciever address.
        If address (Rx or Tx) is None that it will be ignored.

        Parameters
        ----------
        tx: Any
            source address.
        rx: Any
            reciever address.

        Returns
        -------
        Message
            object message instance.
        """
        if tx is not None:
            self._tx = tx
        if rx is not None:
            self._rx = rx
        return self

    @property
    def address(self) -> FieldAddress:
        """
        Returns
        -------
        FieldAddress
            address field instance.
        """
        return self._fields["address"]

    @property
    def data(self) -> FieldData:
        """
        Returns
        -------
        FieldData
            data field instance.
        """
        return self._fields["data"]

    @property
    def data_length(self) -> FieldDataLength:
        """
        Returns
        -------
        FieldDataLength
            data length field instance.
        """
        return self._fields["data_length"]

    @property
    def format_name(self) -> str:
        """
        Returns
        -------
        str
            name of the message format.
        """
        return self._fmt_name

    @property
    def operation(self) -> FieldOperation:
        """
        Returns
        -------
        FieldOperation
            operation field instance.
        """
        return self._fields["operation"]

    @property
    def rx(self):
        return self._rx

    @property
    def slice_length(self):
        return self._slice_length

    @property
    def splitable(self):
        return self._splitable

    @property
    def tx(self):
        return self._tx


class MessageView(MessageBase):

    def hex(self, sep: str = " ", sep_step: int = None) -> str:
        """
        Returns a string of hexademical numbers from the fields content.

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

        See Also
        --------
        Field.hex: return field content hex string.
        """
        fields_hex = []
        for field in self:
            field_hex = field.hex(sep=sep, sep_step=sep_step)
            if field_hex != "":
                fields_hex.append(field_hex)
        return sep.join(fields_hex)

    def to_bytes(self) -> bytes:
        """
        Returns
        -------
        bytes
            joined fields contents.
        """
        return b"".join(
            bytes(field) for field in self._fields.values()
        )

    def unpack(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            unpacked joined fields content.
        """
        unpacked = np.array([])
        for field in self:
            unpacked = np.append(unpacked, field.unpack())
        return unpacked

    @staticmethod
    def _format_address(address: Any) -> str:
        match address:
            case str():
                return address
            case (str() as ip, int() as port):
                return f"{ip}:{port}"
            case _:
                return str(address)

    @property
    def rx_str(self):
        return self._format_address(self._rx)

    @property
    def tx_str(self):
        return self._format_address(self._tx)

    def __bytes__(self) -> bytes:
        """Returns joined fields content."""
        return self.to_bytes()

    def __getitem__(self, name: str) -> Field:
        """
        Returns a field instance by field name.

        Parameters
        ----------
        name : str
            field name.

        Returns
        -------
        Field
            field instance.
        """
        return self._fields[name]

    def __iter__(self):
        """
        Iteration by fields.

        Yields
        -------
        Field
            field instance.
        """
        for field in self._fields.values():
            yield field

    def __len__(self) -> int:
        """Returns length of the message in bytes."""
        return len(self.to_bytes())

    def __repr__(self):
        """Returns string representation of the message."""
        fields_repr = []
        for name, field in self._fields.items():
            if field.words_count:
                fields_repr.append((name, str(field))) # danger with huge fields
            else:
                fields_repr.append((name, "EMPTY"))
        fields_repr = ", ".join(
            f"{name}={field}" for name, field in fields_repr
        )
        return f"<{self.__class__.__name__}({fields_repr}), " \
               f"from={self.tx_str}, to={self.rx_str}>"

    def __str__(self) -> str:
        """Returns fields converted to string."""
        return " ".join(str(field) for field in self if str(field) != "")


class Message(MessageView):
    """
    Represents a message for communication between devices.

    Parameters
    ----------
    format_name: str
        name of the message format.
    splitable: bool
        shows that the message can be divided by the data.
    slice_length: int
        max length of the data in one slice.
    """

    def __init__(
            self,
            format_name: str = "default",
            splitable: bool = False,
            slice_length: int = 1024
    ):
        MessageView.__init__(
            self,
            format_name=format_name,
            splitable=splitable,
            slice_length=slice_length
        )

    @overload
    def configure(
            self,
            address: FieldSetter = None,
            data: FieldSetter = None,
            data_length: FieldSetter = None,
            operation: FieldSetter = None,
            **fields: FieldSetter
    ):
        ...

    def configure(self, **fields: FieldSetter):
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

        next_start_byte = 0
        self._fields.clear()

        for name, setter in fields.items():
            field = self._get_field(name, next_start_byte, setter)
            self._fields[name] = field

            if field.expected <= 0:
                break
            next_start_byte = field.start_byte + \
                field.expected * field.bytesize

        self._configured_fields = fields
        self._configured = True
        return self

    def extract(self, message: bytes):
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
        if not self._configured:
            raise NotConfiguredMessageError(self.__class__.__name__)

        for field in self._fields.values():
            field.extract(message)
        self._validate_content()
        return self

    def get_same_instance(self):
        """
        Get the same class as the current object, initialized with
        the same arguments.

        Also configure fields in the message, but with empty content.

        Returns
        -------
        Message
            new class instance.
        """
        return self.__class__(**self._kwargs) \
            .configure(**self._configured_fields)

    @overload
    def set(
            self,
            address: Content,
            data: Content,
            data_length: Content,
            operation: Content,
            **fields: Content
    ):
        ...

    def set(self, **fields: Content):
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
        if not self._configured:
            raise NotConfiguredMessageError(self.__class__.__name__)

        for name, content in fields.items():
            self[name].set(content)
        self._validate_content()
        return self

    def split(self):
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
                if self.data_length.units == FieldDataLength.WORDS:
                    msg.data.set(self.data[start:end])
                elif self.data_length.units == FieldDataLength.BYTES:
                    msg.data.set(self.data.content[start:end])
                else:
                    raise TypeError('Unsuppoted data units')
                msg.data_length.update(msg.data)
            msg.address.set(
                self.address[0] + i_part * self._slice_length)
            msg.set_addresses(self._tx, self._rx)
            yield msg

    def _get_field(
            self, name: str, start_byte: int, setter: FieldSetter
    ) -> Field:
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
        Field
            field instance.
        """
        if name in self.REQ_FIELDS:
            return self.REQ_FIELDS[name](
                self._fmt_name,
                start_byte=start_byte,
                **setter.kwargs
            )
        return self.SPECIAL_FIELDS.get(setter.special, Field)(
            self._fmt_name,
            name,
            start_byte=start_byte,
            **setter.kwargs
        )

    def _validate_content(self) -> None:
        """Validate content."""
        for field in self:
            if not (field.words_count or field.may_be_empty):
                raise MessageContentError(
                    self.__class__.__name__, field.name, "field is empty"
                )

        oper, dlen = self.operation, self.data_length
        if oper.base == "w" and dlen[0] != dlen.calculate(self.data):
            raise MessageContentError(
                self.__class__.__name__, dlen.name, "invalid length"
            )

    def __add__(self, other):
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
                if other.format_name != self._fmt_name:
                    raise TypeError(
                        "messages have different formats: %s != %s" % (
                            other.format_name, self._fmt_name
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
        self.data_length.update(self.data)
        return self

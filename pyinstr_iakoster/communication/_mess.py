from typing import Any, overload

import numpy as np
import numpy.typing as npt

from ._fields import (
    Content,
    Field,
    FieldSingle,
    FieldStatic,
    FieldAddress,
    FieldData,
    FieldDataLength,
    FieldOperation,
    FloatWordsCountError,
    PartialFieldError,
)
from ..exceptions import (
    NotConfiguredMessageError
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
    "FloatWordsCountError",
    "PartialFieldError",
]


class FieldSetter(object):

    def __init__(
            self,
            *args: Any,
            special: str = None,
            **kwargs: Any,
    ):
        self.args = args
        self.special = special
        self.kwargs = kwargs

    @classmethod
    def base(
            cls,
            expected: int,
            fmt: str,
            content: Content = b"",
            info: dict[str, Any] = None,
    ):
        """For classical field"""
        return cls(expected, fmt, content=content, info=info)

    @classmethod
    def single(
            cls,
            fmt: str,
            content: Content = b"",
            info: dict[str, Any] = None
    ):
        return cls(fmt, special="single", content=content, info=info)

    @classmethod
    def static(
            cls,
            fmt: str,
            content: Content,
            info: dict[str, Any] | None = None,
    ):
        return cls(fmt, content, special="static", info=info)

    @classmethod
    def address(
            cls,
            fmt: str,
            content: Content = b"",
            info: dict[str, Any] | None = None
    ):
        return cls(fmt, content=content, info=info)

    @classmethod
    def data(
            cls,
            expected: int,
            fmt: str,
            content: Content = b"",
            info: dict[str, Any] | None = None
    ):
        return cls(expected, fmt, content=content, info=info)

    @classmethod
    def data_length(
            cls,
            fmt: str,
            content: Content = b"",
            units: int = FieldDataLength.BYTES,
            additive: int = 0,
            info: dict[str, Any] | None = None
    ):
        return cls(
            fmt, content=content, units=units, additive=additive, info=info
        )

    @classmethod
    def operation(
            cls,
            fmt: str,
            desc_dict: dict[str, int] = None,
            content: Content | str = b"",
            info: dict[str, Any] | None = None
    ):
        return cls(fmt, desc_dict=desc_dict, content=content, info=info)


class Message(object):
    """
    Represents a message for communication between devices.

    Parameters
    ----------
    format_name: str
        name of the message format.
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
    _addr: FieldAddress
    _data: FieldData
    _dlen: FieldDataLength
    _oper: FieldOperation

    def __init__(
            self,
            format_name: str = "default"
    ):
        self._fmt_name = format_name
        self._fields: dict[str, Field] = {}
        self._configured = False
        self._tx, self._rx = None, None

        self._args = ()
        self._kwargs = dict(
            format_name=format_name,
        )
        self._configured_fields: dict[str, FieldSetter] = {}

    def clear_addresses(self) -> None:
        """Set addresses to None."""
        self._tx, self._rx = None, None

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

        req_attrs = {
            "address": "_addr",
            "data": "_data",
            "data_length": "_dlen",
            "operation": "_oper"
        }
        next_start_byte = 0
        self._fields.clear()

        for name, setter in fields.items():
            field = self._get_field(name, next_start_byte, setter)
            self._fields[name] = field
            if name in self.REQ_FIELDS:
                object.__setattr__(self, req_attrs[name], field)

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

    def get_instance(self, *args: Any, **kwargs: Any):
        """
        Get the same class as the current object, initialized with
        the specified arguments.

        Parameters
        ----------
        *args: Any
            initial arguments.
        **kwargs: Any
            initial keywords arguments.

        Returns
        -------
        Message
            new class instance.
        """
        return self.__class__(*args, **kwargs)

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
        return self.__class__(*self._args, **self._kwargs) \
            .configure(**self._configured_fields)

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

    @overload
    def set_fields_content(
            self,
            address: Content,
            data: Content,
            data_length: Content,
            operation: Content,
            **fields: Content
    ):
        ...

    def set_fields_content(self, **fields: Content):
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

        for name, content in fields:
            self[name].set(content)
        self._validate_content()
        return self

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

    def to_bytes(self) -> bytes:
        """
        Returns
        -------
        bytes
            joined fields contents.
        """
        return b"".join(bytes(field) for field in self)

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
                start_byte,
                *setter.args,
                **setter.kwargs
            )
        return self.SPECIAL_FIELDS.get(setter.special, Field)(
            self._fmt_name,
            name,
            start_byte,
            *setter.args,
            **setter.kwargs
        )

    def _validate_content(self) -> None:
        """Validate content."""
        ...

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
    def address(self) -> FieldAddress:
        """
        Returns
        -------
        FieldAddress
            address field instance.
        """
        return self._addr

    @property
    def data(self) -> FieldData:
        """
        Returns
        -------
        FieldData
            data field instance.
        """
        return self._data

    @property
    def data_length(self) -> FieldDataLength:
        """
        Returns
        -------
        FieldDataLength
            data length field instance.
        """
        return self._dlen

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
        return self._oper

    @property
    def rx(self):
        return self._rx

    @property
    def rx_str(self):
        return self._format_address(self._rx)

    @property
    def tx(self):
        return self._tx

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
        fields_repr = ", ".join(f"{name}={field}" for name, field in fields_repr)
        return f"<{self.__class__.__name__}({fields_repr}), from={self.tx_str}, to={self.rx_str}>"

    def __str__(self) -> str:
        """Returns fields converted to string."""
        return " ".join(str(field) for field in self if str(field) != "")


# class Message(object):
#     """
#     Родительский класс сообщения
#     """
#
#
#     _max_data_len = 256
#     _cuttable_data = True
#
#     _package_format = ''
#     _module_name = ''
#
#     def __init__(self, *args, **kwagrs):
#         self._fields = {}
#         self._init_args = args
#         self._init_kwargs = kwagrs
#         self._from_addr = None
#         self._to_addr = None
#
#     def _exc_if_none_in_field(self) -> None:
#         """
#         Вызывает исключение, если какое-либо из полей
#         (кроме поля data) пустое
#
#         :return: None
#         """
#         for field in self.fields:
#             if field.content == b'' and field.name != 'data':
#                 raise ValueError(f'field \'{field.name}\' is unfilled')
#
#     def split(self):
#         """
#         разделить сообщение на части, обусловленными максимальной
#         длиной данных в одном сообщении (max_data_len)
#
#         :return: сообщение
#         """
#         if not self._cuttable_data:
#             raise TypeError(
#                 f'{self.__class__.__name__} cannot be split by data')
#
#         parts_count = int(np.ceil(
#             self.data_len.unpack() / self._max_data_len))
#         for i_part in range(parts_count):
#             data_len = self.data_len.unpack() - i_part * self._max_data_len \
#                 if i_part == parts_count - 1 else self._max_data_len
#             msg_part = deepcopy(self)
#
#             if msg_part.oper.desc_base == 'r':
#                 msg_part.data_len.set_content(data_len)
#
#             elif msg_part.oper.desc_base == 'w':
#                 start = i_part * self._max_data_len
#                 end = start + data_len
#
#                 if self.data_len.data_dim == 'words':
#                     msg_part.data.set_content(self.data[start:end])
#                 elif self.data_len.data_dim == 'bytes':
#                     msg_part.data.set_content(self.data.content[start:end])
#                 else:
#                     raise TypeError('Unsuppoted data dimension')
#                 msg_part.data_len.upd_content_by_data(msg_part.data)
#
#             msg_part.addr.set_content(
#                 self.addr.unpack() + i_part * self._max_data_len)
#             msg_part.set_from_to_addresses(self._from_addr, self._to_addr)
#
#             yield msg_part
#
#     @property
#     def max_data_len(self) -> int:
#         """
#         :return: максимальная длина данных в одиночном сообщении
#         при разбиении
#         """
#         return self._max_data_len
#
#     @property
#     def cuttable_data(self) -> bool:
#         """
#         :return: указатель на то, можно ли делить данное сообщение
#         """
#         return self._cuttable_data
#
#     def __add__(self, add_data):
#         """
#         Если передаваемый аргумент есть тип bytes,
#         то к полю data прибавляется переданный аргумент
#         (конкатенация справа).
#
#         Если передаваемый аргумент является экземпляром
#         того же класса, то проверяется идентичность
#         имени формата пакетов и модуля и затем содержание
#         поля data прибавляется к полю data текущего экземпляра.
#
#         В обоих случаях перед записью проверяется, что новые данные
#         будут содержать целое количество слов в формате текущего
#         экземпляра
#
#         :param add_data: сообщение класса Message или типа bytes
#         :return: self
#         """
#         match add_data:
#             case Message(_package_format=self._package_format,
#                          _module_name=self._module_name):
#                 add_data = add_data.data.content
#             case bytes():
#                 pass
#             case Message() | bytes():
#                 raise TypeError(
#                     'there is a diffierens in package format or '
#                     'module name between the terms')
#             case _:
#                 raise TypeError(
#                     f'{type(add_data)} cannot be added to the message')
#
#         self.data.set_content(self.data.content + add_data)
#         self.data_len.upd_content_by_data(self.data)
#
#         return self
#

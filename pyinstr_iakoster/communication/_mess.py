from typing import Any, overload

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

        self._configured = True
        return self

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
            self class instance

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

    def _validate_content(self) -> None:
        """Validate content."""
        ...

    def _get_field(
            self, name: str, start_byte: int, setter: FieldSetter
    ) -> Field:
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

    @property
    def address(self) -> FieldAddress:
        return self._addr

    @property
    def data(self) -> FieldData:
        return self._data

    @property
    def data_length(self) -> FieldDataLength:
        return self._dlen

    @property
    def format_name(self) -> str:
        return self._fmt_name

    @property
    def operation(self) -> FieldOperation:
        return self._oper

    def __getitem__(self, name: str) -> Field:
        return self._fields[name]

    def __iter__(self):
        for field in self._fields.values():
            yield field


# class Message(object):
#     """
#     Родительский класс сообщения
#     """
#
#     _req_fields_dict = {'addr': MessageFieldAddr,
#                         'data_len': MessageFieldDataLen,
#                         'oper': MessageFieldOper,
#                         'data': MessageFieldData}
#     _fields: dict[str, MessageField]
#
#     _max_data_len = 256
#     _cuttable_data = True
#
#     addr: MessageFieldAddr
#     data_len: MessageFieldDataLen
#     oper: MessageFieldOper
#     data: MessageFieldData
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
#     def get_instance(self, *args, **kwargs):
#         """
#         Возвращает тот же класс, что и текущий.
#
#         :param args: аргументы нового экземпляра класса.
#         :param kwargs: кварги нового экземпляра класса.
#         :return: новый экземпляр текущего класса.
#         """
#         return self.__class__(*args, **kwargs)
#
#     def get_same_instance(self):
#         """
#         Возвращает тот же класс с теми же настройками,
#         что и текущий.
#
#         :return: новый экземпляр текущего класса.
#         """
#         return self.__class__(*self._init_args, **self._init_kwargs)
#
#     @overload
#     def set_fields_content(
#             self, addr: _mf_content_types = None,
#             oper: _mf_content_types = None,
#             data_len: _mf_content_types = None,
#             data: _mf_content_types = None,
#             **kwargs: _mf_content_types
#     ) -> None:
#         ...
#
#     def set_fields_content(self, **kwargs: _mf_content_types):
#         """
#         Установить содержание полей
#         """
#         for field_name, field_value in kwargs.items():
#             self[field_name].set_content(field_value)
#         self._exc_if_none_in_field()
#         return self
#
#     def extract_from(self, message: bytes) -> None:
#         """
#         Передает массив байт в поле, где происходит
#         извлечение содержания
#
#         :param message: сообщение
#         :return: None
#         """
#         for field in self._fields.values():
#             field.extract_from(message)
#         self._exc_if_none_in_field()
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
#     def inbytes(self) -> bytes:
#         """
#         :return: сообщение, преобразованное в тип bytes
#         """
#         return b''.join(
#             bytes(field) for field in
#             self._fields.values()
#         )
#
#     def unpack(self) -> list[int]:
#         """
#         распаковать соодщение в слова
#
#         :return: список слов
#         """
#         return [word for field in self._fields.values() for word in field]
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
#     def hex(self, sep: str = ' ', sep_step: int = None) -> str:
#         """
#         Перевести bytes в hex-строку
#
#         :param sep: разделитель между каждым байтом
#         :param sep_step: шаг разделителя в байтах
#         :return: hex-строка
#         """
#         raw_hex = []
#         for field in self._fields.values():
#             raw_hex.append(field.hex(
#                 sep=sep, sep_step=sep_step))
#         return sep.join(raw_hex)
#
#     def set_from_to_addresses(self, from_, to) -> None:
#         self.set_from_address(from_)
#         self.set_to_address(to)
#
#     def set_from_address(self, address) -> None:
#         self._from_addr = address
#
#     def set_to_address(self, address) -> None:
#         self._to_addr = address
#
#     @staticmethod
#     def _fmt_address(address) -> str:
#         if isinstance(address, tuple | list) and len(address) == 1:
#             address = address[0]
#
#         if isinstance(address, tuple | list):
#             return ':'.join(map(str, address))
#         else:
#             return str(address)
#
#     @property
#     def package_format(self) -> str:
#         """
#         Returns the name of package format to which
#         the message belongs
#         """
#         return self._package_format
#
#     @property
#     def module_name(self) -> str:
#         """
#         Returns the name or str-code of device module
#         to which the message belongs
#         """
#         return self._module_name
#
#     @property
#     def fields(self) -> tuple[MessageField]:
#         """
#         :return: кортеж с текущими полями в сообщении
#         """
#         return tuple(self._fields.values())
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
#     @property
#     def from_address(self):
#         """
#         :return: адрес отправителя
#         """
#         return self._from_addr
#
#     @property
#     def to_address(self):
#         """
#         :return: адрес получателя
#         """
#         return self._to_addr
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
#     def __bytes__(self) -> bytes:
#         """
#         :return: сообщение как массив байтов
#         """
#         return self.inbytes()
#
#     def __getitem__(self, field_name: str) -> MessageField:
#         """
#         :param field_name: имя поля
#         :return: поле сообщения
#         """
#         return self._fields[field_name]
#
#     def __iter__(self):
#         """
#         Итерирует по словам в сообщении
#
#         :return: слово
#         """
#         for word in self.unpack():
#             yield word
#
#     def __len__(self) -> int:
#         """
#         :return: количество байт в сообщении
#         """
#         return len(self.inbytes())
#
#     def __str__(self):
#         """
#         :return: hex-строка сообщения
#         """
#         return ' '.join(str(field) for field in self._fields.values())
#
#     def __repr__(self):
#         fields_str = ', '.join(
#             f'{name}={value}' for name, value in self._fields.items())
#         from_to_str = '{}->{}'.format(
#             self._fmt_address(self._from_addr),
#             self._fmt_address(self._to_addr))
#         return f'<{self.__class__.__name__}' \
#                f'({fields_str}), from_to={from_to_str}>'

import struct
from copy import deepcopy
from typing import Any, Sized, Iterable, overload, Type, NewType, SupportsBytes
from collections import namedtuple

import numpy as np
import numpy.typing as npt


from ..exceptions import (
    FloatWordsCountError,
    PartialFieldError,
)


__all__ = [
    "FieldBase"
]


Content = (
        bytes | bytearray | int | float | Iterable |
        SupportsBytes | np.number | npt.NDArray
)


class FieldBase(object):
    """
    Represents a single field of a Message

    Parameters
    ----------
    :param package_format: имя формата пакетов
    :param module_name: имя модуля
    :param name: имя поля
    :param start_byte: стартовый байт поля в сообщении
    :param expected: ожидаемое количество слов в поле
        (если == -1, то от стартового байта и до конца сообщения)
    :param fmt: формат содержания поля
    :param content: соержание поля
    """

    def __init__(
            self,
            format_name: str,
            name: str,
            info: dict[str, Any],
            *,
            start_byte: int,
            expected: int,
            fmt: str,
            content: Content
    ):
        self._fmt_name = format_name
        self._name = name
        self._info = info
        self._st_byte = start_byte
        self._exp = expected
        self._fmt = fmt
        self._content = b''

        self._word_bsize = struct.calcsize(self._fmt)
        if expected > 0:
            self._fin = True
            self._end_byte = start_byte + self._word_bsize * expected
            self._slice = slice(start_byte, self._end_byte)
        else:
            self._fin = False
            self._end_byte = np.inf
            self._slice = slice(start_byte, None)

        self.set_content(content)

    def _convert_content(self, content: Content) -> bytes:
        """
        Convert content to bytes via `fmt` or `__bytes__`.

        Parameters
        ----------
        content: Content
            content for converting.

        Returns
        -------
        bytes
            content converted to bytes.
        """

        if isinstance(content, bytes):
            converted = content
        elif isinstance(content, bytearray):
            converted = bytes(content)
        elif isinstance(content, npt.NDArray):
            converted = content.astype(self._fmt).tobytes()
        elif isinstance(content, Iterable):
            converted = np.array(content, dtype=self._fmt).tobytes()
        elif np.issubdtype(type(content), np.number):
            converted = struct.pack(self._fmt, content)
        else:
            converted = bytes(content)

        return converted

    def _validate_content(self, content: bytes = None) -> bytes:
        """
        Проверить на корректность количество слов в байтах

        :param content: содержание поля
        :return: None
        """
        if content is None:
            content = self._content
        if content == b"":
            return b""

        if len(content) % self._word_bsize != 0:
            raise FloatWordsCountError(
                self.__class__.__name__,
                self._exp,
                len(content) / self._word_bsize
            )

        # Similary to self._exp > 0 and len(content) / self._exp != 1
        if 0 < self._exp != len(content) / self._exp:
            raise PartialFieldError(
                self.__class__.__name__, len(content) / self._exp
            )

        return content

    def set_content(self, content: Content) -> None:
        """
        Установить содержание поля

        :param content: содержание поля. Если аргумент не является типом bytes,
            то происходит преобразование в bytes согласно аттрибуту fmt
        :return: None
        """

        if not isinstance(content, bytes):
            content = self._convert_content(content)
        self._content = self._validate_content(content=content)

    def extract_from(self, message: bytes) -> None:
        """
        Извлекает из принятого в метод сообщения поле
        от start_byte до word_length * word_count.

        Если word_count < 1, то в поле записывается все от
        start_byte до конца принятого сообщения

        :param message: сообщение, из которого извлечь поле
        :return: None
        """
        self._content = self._validate_content(message[self._slice])

    def unpack(self, fmt: str = None) -> list[int | float]:
        """
        Возвращает содержание поля, представленное как fmt

        :param fmt: формат представления. Если аргумент == None,
            то fmt берется из аттрибута класса, указанного
            при создании
        :return: list, где каждое значение есть слово
        """
        if fmt is None:
            fmt = self._fmt
        return [word for word, in struct.iter_unpack(fmt, self._content)]

    def hex(self, sep: str = ' ', sep_step: int = None) -> str:
        """
        Перевести bytes в hex-строку

        :param sep: разделитель между каждым байтом
        :param sep_step: шаг разделителя в байтах
        """
        if sep_step is None:
            sep_step = self._word_bsize
        return self._content.hex(sep=sep, bytes_per_sep=sep_step)

    @property
    def package_format(self) -> str:
        """Returns the name of package format to which the field belongs."""
        return self._fmt_name

    @property
    def name(self) -> str:
        """Returns the name of the massage field."""
        return self._name

    @property
    def start_byte(self) -> int:
        """Returns the number of byte in the message from which
        the field starts."""
        return self._st_byte

    @property
    def bytesize(self) -> int:
        """Returns the length of the one word in bytes."""
        return self._word_bsize

    @property
    def expected(self) -> int:
        """Returns the expected count of words."""
        return self._exp

    @property
    def words_count(self) -> int:
        """Returns the length of the field in words."""
        return len(self._content) // self._word_bsize

    @property
    def fmt(self) -> str:
        """Returns the converion format from bytes to number."""
        return self._fmt

    @property
    def content(self) -> bytes:
        """Returns field content."""
        return self._content

    @property
    def field_class(self):
        """Returns field class."""
        return self.__class__

    def __bytes__(self) -> bytes:
        """
        :return: содержание поля
        """
        return self._content

    def __getitem__(self, word_index):
        """
        Возвращает из поля слово или срез слов,
        распакованных заданным при инициализации
        форматом
        """
        return self.unpack()[word_index]

    def __iter__(self):
        """
        Итерирование по словам, преобразованным согласно fmt

        :return: слово
        """
        for word in self.unpack():
            yield word

    def __len__(self) -> int:
        """
        :return: длина содержания
        """
        return len(self._content)

    def __str__(self):
        """
        Возвращает строку, с представлением содержания
        в читаемом формате.

        При конвертации убираются левые незначащие нули
        и устанавливается разделителем пробел через каждый
        байт

        :return: hex-строка содержания
        """

        hex_words = []
        for i_byte in range(
                0, int(self.words_count * self._word_bytesize),
                self._word_bytesize):

            val = self._content[i_byte:i_byte + self._word_bytesize] \
                .hex().lstrip('0')
            if len(val) == 0:
                val = '0'
            hex_words.append(val)

        return ' '.join(hex_words)

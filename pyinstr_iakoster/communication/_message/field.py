from __future__ import annotations
import struct
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    SupportsBytes,
    Generator,
    Protocol,
    Callable,
    runtime_checkable
)

import numpy as np
import numpy.typing as npt

from pyinstr_iakoster.core import Code
from pyinstr_iakoster.exceptions import (
    FieldContentError
)
if TYPE_CHECKING:
    from .message import Message


__all__ = [
    "ContentType",
    "Field",
    "FieldSetter",
    "AddressField",
    "CrcField",
    "DataField",
    "DataLengthField",
    "OperationField",
    "SingleField",
    "StaticField",
    "ResponseField",
    "FieldType",
    "FieldContentError"
]


ContentType = (
        bytes | bytearray | int | float | Iterable |
        SupportsBytes | np.number | npt.NDArray
)


class BaseField(object):
    """
    Represents a basic class for single field of a Message.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    info: dict of {str, Any}
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    expected: int
        expected number of words in the field. If equal to -1, from
        the start byte to the end of the message.
    may_be_empty: bool
        if True then field can be empty in a message.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    content: bytes
        field content in bytes.
    parent: Message or None
        parent message.
    """

    def __init__(
            self,
            format_name: str,
            name: str,
            *,
            start_byte: int,
            expected: int,
            may_be_empty: bool,
            fmt: str,
            info: dict[str, Any],  # todo: unused parameter
            content: bytes,
            default: ContentType,
            parent: Message,
    ):
        self._fmt_name = format_name
        self._name = name
        self._info = info
        self._exp = expected
        self._may_be_empty = may_be_empty
        self._fmt = fmt
        self._content = content
        self._def = default
        self._parent = parent

        self._word_bsize = struct.calcsize(self._fmt)
        self._fin = expected > 0
        self._slice = slice(
            start_byte,
            start_byte + self._word_bsize * expected if self._fin else None
        )

    @property
    def bytesize(self) -> int:
        """The length of the one word in bytes."""
        return self._word_bsize

    @property
    def content(self) -> bytes:
        """The field content."""
        return self._content

    @property
    def default(self) -> bytes:
        """Default field content"""
        return self._def

    @property
    def expected(self) -> int:
        """The expected count of words."""
        return self._exp

    @property
    def finite(self):
        return self._fin

    @property
    def fmt(self) -> str:
        """The converion format."""
        return self._fmt

    @property
    def format_name(self) -> str:
        """The name of package format to which the field belongs."""
        return self._fmt_name

    @property
    def info(self) -> dict[str, Any]:
        """Additional information about the field."""
        return self._info

    @property
    def may_be_empty(self) -> bool:
        """May be empty content in the field"""
        return self._may_be_empty

    @property
    def name(self) -> str:
        """The name of the massage field."""
        return self._name

    @property
    def parent(self) -> Message | None:
        return self._parent

    @property
    def slice(self):
        """The range of bytes from the message belonging to the field"""
        return self._slice

    @property
    def start_byte(self) -> int:
        """The number of byte in the message from which the field starts."""
        return self._slice.start

    @start_byte.setter
    def start_byte(self, start: int) -> None:
        """Set the number of a start byte of a field in a message."""
        self._slice = slice(start, self._slice.stop)

    @property
    def stop_byte(self) -> int | None:
        """The number of byte in the message to which the field stops."""
        return self._slice.stop

    @stop_byte.setter
    def stop_byte(self, stop: int | None) -> None:
        """Set the numbet of a stop byte of a field in a message."""
        self._slice = slice(self._slice.start, stop)

    @property
    def words_count(self) -> int:
        """The length of the field in words."""
        return len(self._content) // self._word_bsize

    def __bytes__(self) -> bytes:
        """Returns field content"""
        return self._content

    def __len__(self) -> int:
        """Returns the length of the content in bytes"""
        return len(self._content)


class Field(BaseField):
    """
    Represents a general field of a Message.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    expected: int
        expected number of words in the field. If equal to -1, from
        the start byte to the end of the message.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    default: Content, default=b""
        field content.
    may_be_empty: bool, default=False
        if True then field can be empty in a message.
    parent: Message or None
            parent message.

    See Also
    --------
    FieldBase: parent class.
    """

    def __init__(
            self,
            format_name: str,
            name: str,
            *,
            start_byte: int,
            expected: int,
            fmt: str,
            info: dict[str, Any] = None,
            default: ContentType = b"",
            may_be_empty: bool = False,
            parent: Message = None,
    ):
        if info is None:
            info = {}
        BaseField.__init__(
            self,
            format_name,
            name,
            start_byte=start_byte,
            expected=expected,
            may_be_empty=may_be_empty,
            fmt=fmt,
            info=info,
            content=b"",
            default=b"",
            parent=parent
        )
        if not isinstance(default, bytes) or len(default):
            self.set(default)
            self._def = self._content

    def get_setter(self) -> FieldSetter:
        """
        Returns
        -------
        FieldSetter
            field setter for a message configure method.

        See Also
        --------
        Message.get_same_instance: where it used.
        """
        return FieldSetter.base(
            expected=self._exp,
            fmt=self._fmt,
            default=self.unpack_default(),
            info=self._info,
            may_be_empty=self._may_be_empty
        )

    def set(self, content: ContentType) -> None:
        """
        Set the field content.

        If the received content is not of type bytes, then there will be
        an attempt to convert the received variable into type bytes,
        otherwise an exception will be raised.

        Parameters
        ----------
        content: Content type
            new field content.

        See Also
        --------
        _validate_content: method of validating new content.
        """
        self._content = self._validate_content(self._convert_content(content))

    def extract(self, message: bytes) -> None:
        """
        Extract new content from a message in range self.slice.

        Extracts a field from start_byte to bytesize * expected from
        the message received in the method. If expected < 1, it writes
        everything from start_byte to the end of the received message.

        Parameters
        ----------
        message: bytes
            incoming message.

        Raises
        ------
        ValueError
            if message is empty.

        See Also
        --------
        set_content: method of setting new field content.
        """
        if len(message) == 0:
            raise ValueError(
                "Unable to extract because the incoming message is empty"
            )
        self.set(message[self._slice])

    def hex(self, sep: str = " ", sep_step: int = None) -> str:
        """
        Returns a string of hexadecimal numbers from the content.

        Parameters
        ----------
        sep: str
            separator between bytes.
        sep_step: int
            separator step. If None equals bytesize.

        Returns
        -------
        str
            hex string.
        """
        if sep_step is None:
            sep_step = self._word_bsize
        return self._content.hex(sep=sep, bytes_per_sep=sep_step)

    def reset_to_default(self) -> None:
        """Set field content to default."""
        self._content = self._def

    def unpack(self, fmt: str = None) -> npt.NDArray:
        """
        Returns the content of the field unpacked in fmt.

        If fmt is None, that is taken from an instance of the class.

        Parameters
        ----------
        fmt: str
            format for unpacking.

        Returns
        -------
        numpy.ndarray
            an array of unpacked bytes.
        """
        return self._unpack_bytes(self._content, fmt=fmt)

    def unpack_default(self, fmt: str = None) -> npt.NDArray:
        """
        Returns the default content unpacked in fmt.

        If fmt is None, that is taken from an instance of the class.

        Parameters
        ----------
        fmt: str
            format for unpacking.

        Returns
        -------
        numpy.ndarray
            an array of unpacked bytes.
        """
        return self._unpack_bytes(self._def, fmt=fmt)

    def _convert_content(self, content: ContentType) -> bytes:
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
        elif isinstance(content, np.ndarray):
            converted = content.astype(self._fmt).tobytes()
        elif isinstance(content, Iterable):
            converted = np.array(content, dtype=self._fmt).tobytes()
        elif np.issubdtype(type(content), np.number):
            converted = struct.pack(self._fmt, content)
        else:
            converted = bytes(content)

        return converted

    def _unpack_bytes(self, bytes_: bytes, fmt: str = None) -> npt.NDArray: # todo: add complex fmt (e.g. >HH)
        """
        Returns bytes unpacked in fmt.

        If fmt is None, that is taken from an instance of the class.

        Parameters
        ----------
        bytes_: bytes
            bytes for unpacking.
        fmt: str
            format for unpacking.

        Returns
        -------
        numpy.ndarray
            an array of unpacked bytes.
        """
        if fmt is None:
            fmt = self._fmt
        return np.frombuffer(bytes_, dtype=fmt)

    def _validate_content(
            self, content: bytes = None, exp: int = None
    ) -> bytes:
        """
        Validate field content.

        If the content is None, checks the content from the class instance.

        Parameters
        ----------
        content: bytes, default=None
            content for validating.
        exp: int, default=None
            expected words in content.

        Returns
        -------
        bytes
            content.

        Raises
        ------
        FieldContentError
            if the number of words in the content is not an integer.
            if the length of the content is less than expected.
        """
        if content is None:
            content = self._content
        if content == b"":
            return content
        if exp is None:
            exp = self._exp

        if len(content) % self._word_bsize != 0:
            words_count = len(content) / self._word_bsize
            raise FieldContentError(
                self.__class__,
                exp,
                words_count,
                clarification=(
                        "not integer count of words "
                        "(expected %d, got %.1f)" % (exp, words_count)
                )
            )

        # Similary to self._exp > 0 and
        # len(content) / self._word_bsize != self._exp
        if 0 < exp != len(content) / self._word_bsize:
            fill_ratio = len(content) / (exp * self._word_bsize)
            raise FieldContentError(
                self.__class__,
                fill_ratio,
                clarification="fill ratio - %.1f" % fill_ratio
            )

        return content

    def __getitem__(self, word_index: int | slice) -> int | float | npt.NDArray:
        """
        Returns from the field a word or a slice of words unpacked in fmt.

        Parameters
        ----------
        word_index: int or slice
            requared word(s).

        Returns
        -------
        int of float
            unpacked word(s).
        """
        return self.unpack()[word_index]

    def __iter__(self) -> Generator[int | float, None, None]:
        """
        Iterating over words unpacked in fmt.

        Yields
        ------
        int or float
            unpacked word.
        """
        for word in self.unpack():
            yield word

    def __str__(self) -> str:
        """
        Returns a string representing the content in a readable format.

        When converting, left insignificant zeros are removed and
        a space separator between words.

        Returns
        -------
        str
            content as readable string.
        """
        words = []
        for start in range(0, len(self.content), self.bytesize):
            word = self.content[start:start + self.bytesize]\
                .hex().lstrip("0").upper()
            if not len(word):
                word = "0"
            words.append(word)
        return " ".join(words)

    def __repr__(self) -> str:
        """Returns string representation of the field instance"""
        words_count = self.words_count
        if words_count > 16:
            self_str = "{} ...({})".format(
                " ".join(str(self).split(" ")[:8]), words_count - 8
            )
        else:
            self_str = str(self)
        return f"<{self.__class__.__name__}({self_str}, fmt='{self._fmt}')>"


class SingleField(Field):
    """
    Represents a field of a Message with single word.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    may_be_empty: bool, default=False
        if True then field can be empty in a message.
    parent: Message or None
        parent message.

    Notes
    -----
    The __getitem__ method was disallowed because only one word is expected.

    See Also
    --------
    Field: parent class.
    """

    def __init__(
            self,
            format_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            info: dict[str, Any] = None,
            default: ContentType = b"",
            may_be_empty: bool = False,
            parent: Message = None,
    ):
        Field.__init__(
            self,
            format_name,
            name,
            start_byte=start_byte,
            expected=1,
            fmt=fmt,
            info=info,
            default=default,
            may_be_empty=may_be_empty,
            parent=parent
        )

    def get_setter(self) -> FieldSetter:
        return FieldSetter.single(
            fmt=self._fmt,
            default=self.unpack_default(),
            info=self._info,
            may_be_empty=self._may_be_empty
        )


class StaticField(SingleField):
    """
    Represents a field of a Message with static single word (e.g. preamble).

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    default: Content, default=b""
        field content.
    parent: Message or None
        parent message.

    Notes
    -----
    The __getitem__ method was disallowed because only one word is expected.

    See Also
    --------
    FieldSingle: parent class.
    """

    def __init__(
            self,
            format_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            default: ContentType,
            info: dict[str, Any] | None = None,
            parent: Message = None,
    ):
        SingleField.__init__(
            self,
            format_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            info=info,
            parent=parent
        )
        self.set(default)

    def get_setter(self) -> FieldSetter:
        return FieldSetter.static(
            fmt=self._fmt,
            default=self.unpack_default(),
            info=self._info,
        )

    def set(self, content: ContentType) -> None:
        content = self._convert_content(content)
        if content == b"":
            pass
        elif self._content != b"" and content != self._content:
            raise ValueError(
                "The current content of the static field is different from "
                "the new content: %r != %r" % (self._content, content)
            )
        else:
            self._content = self._validate_content(content)
            self._def = self._content


class AddressField(SingleField):
    """
    Represents a field of a Message with address.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    parent: Message or None
        parent message.

    See Also
    --------
    FieldSingle: parent class.
    """

    def __init__(
            self,
            format_name: str,
            *,
            start_byte: int,
            fmt: str,
            info: dict[str, Any] | None = None,
            parent: Message = None,
    ):
        SingleField.__init__(
            self,
            format_name,
            "address",
            start_byte=start_byte,
            fmt=fmt,
            info=info,
            parent=parent
        )

    def get_setter(self) -> FieldSetter:
        return FieldSetter.address(
            fmt=self._fmt,
            info=self._info,
        )


class CrcField(SingleField):
    """
    Represents a field of a Message with crc value.

    If the field name is 'crc', then Message will automatically update
    the value. The user/developer is responsible for setting the correct
    parameters (e.g. fmt).

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    name: str
        the name of a field.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    algorithm_name: str
        the name of the algorithm by which the crc is counted.
    parent: Message or None
        parent message.

    See Also
    --------
    FieldSingle: parent class.
    """

    def __init__(
            self,
            format_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            info: dict[str, Any] | None = None,
            algorithm_name: str = "crc16-CCITT/XMODEM",
            parent: Message = None,
    ):
        if algorithm_name not in self.CRC_ALGORITHMS:
            raise ValueError("invalid algorithm name: %s" % algorithm_name)

        SingleField.__init__(
            self,
            format_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            info=info,
            parent=parent
        )
        self._alg_name = algorithm_name
        self._alg = self.CRC_ALGORITHMS[algorithm_name]

    def calculate(self, msg: Message) -> int:
        """
        Calculate a crc value of a message with all fields except this
        field instance.

        Parameters
        ----------
        msg: Message
            message.

        Returns
        -------
        int
            crc value.
        """
        return self._alg(
            b"".join(field.content for field in msg if field is not self)
        )

    def update(self) -> None:
        """Update crc value using parent message."""
        self.set(self.calculate(self.parent))

    @staticmethod
    def get_crc16_ccitt_xmodem(content: bytes) -> int:
        """
        Calculate a crc16-CCITT/XMODEM of content.

        Parameters
        ----------
        content

        Returns
        -------
        int
            crc value from 0 to 0xffff.
        """

        crc, poly = 0, 0x1021
        for idx in range(len(content)):
            crc ^= content[idx] << 8
            for _ in range(8):
                crc <<= 1
                if crc & 0x10000:
                    crc ^= poly
            crc &= 0xffff
        return crc

    CRC_ALGORITHMS: dict[str, Callable[[bytes], int]] = {
        "crc16-CCITT/XMODEM": get_crc16_ccitt_xmodem
    }

    @property
    def algorithm(self) -> Callable[[bytes], int]:
        """
        Returns
        -------
        Callable[[bytes], int]
            algorithm function.
        """
        return self._alg

    @property
    def algorithm_name(self) -> str:
        """
        Returns
        -------
        str
            algorithm name.
        """
        return self._alg_name


class DataField(Field):
    """
    Represents a field of a Message with data.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    parent: Message or None
        parent message.

    See Also
    --------
    Field: parent class.
    """

    def __init__(
            self,
            format_name: str,
            *,
            start_byte: int,
            expected: int,
            fmt: str,
            info: dict[str, Any] | None = None,
            parent: Message = None
    ):
        Field.__init__(
            self,
            format_name,
            "data",
            start_byte=start_byte,
            expected=expected,
            fmt=fmt,
            info=info,
            may_be_empty=True,
            parent=parent
        )

    def append(self, content: ContentType) -> None:
        content = self._convert_content(content)
        if self._exp > 0:
            exp = self._exp + len(content) // self._word_bsize
        else:
            exp = None
        self._content = self._validate_content(
            self._content + content, exp=exp
        )
        if self._exp > 0:
            self._exp = exp

    def get_setter(self) -> FieldSetter:
        return FieldSetter.data(
            expected=self._exp,
            fmt=self._fmt,
            info=self._info,
        )


class DataLengthField(SingleField):
    """
    Represents a field of a Message with data length.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    units: int
        data length units. Data can be measured in bytes or words.
    additive: int
        additional value to the length of the data.
    parent: Message or None
        parent message.

    Raises
    ------
    ValueError
        if units not in {BYTES, WORDS}.
    ValueError
        if additive value is not integer or negative.

    See Also
    --------
    SingleField: parent class.
    """

    BYTES = 0x10
    WORDS = 0x11

    def __init__(
            self,
            format_name: str,
            *,
            start_byte: int,
            fmt: str,
            units: int = BYTES,
            additive: int = 0,
            info: dict[str, Any] | None = None,
            parent: Message = None
    ):
        if units not in (self.BYTES, self.WORDS):
            raise ValueError("invalid units: %d" % units)
        if additive < 0 or not isinstance(additive, int):
            raise ValueError(
                "additive number must be integer and positive, "
                f"got {additive}"
            )

        SingleField.__init__(
            self,
            format_name,
            "data_length",
            start_byte=start_byte,
            fmt=fmt,
            info=info,
            parent=parent
        )
        self._units = units
        self._add = additive

    def calculate(self, data: DataField) -> int:
        """
        Calculate data length via data field.

        Parameters
        ----------
        data: DataField
            data field.

        Returns
        -------
        int
            actual length of data field.

        Raises
        ------
        ValueError
            if units not in {BYTES, WORDS}.
        """

        if self._units == self.BYTES:
            return len(data) + self._add
        elif self._units == self.WORDS:
            return data.words_count + self._add
        else:
            raise ValueError(f"invalid units: {self._units}")

    def get_setter(self) -> FieldSetter:
        return FieldSetter.data_length(
            fmt=self._fmt,
            units=self._units,
            additive=self._add,
            info=self._info,
        )

    def update(self) -> None:
        """
        Update data length content via parent message.

        Raises
        ------
        ValueError
            if units not in {BYTES, WORDS}.
        """
        if self._parent is None:
            raise ValueError("There is no parent message")
        self.set(self.calculate(self._parent.data))

    @property
    def units(self) -> int:
        """Data length units."""
        return self._units

    @property
    def additive(self) -> int:
        """Additional value to the data length."""
        return self._add


class OperationField(SingleField):
    """
    Represents a field of a Message with operation (e.g. read).

    Field operation contains operation codes:
        * READ = "r" -- read operation;
        * WRITE = "w" -- write operation;
        * ERROR = "e" -- error operation.
    Operation codes are needed to compare the base of the operation
    (the first letter of desc) when receiving a message and
    generally to understand what operation is written in the message.

    To work correctly, the first letter of keys in desc_dict must be
    one of {'r', 'w', 'e'} (see examples). If the dictionary is None,
    the standard value will be assigned {'r': 0, 'w': 1, 'e': 2}.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    info: dict of {str, Any}, optional
        additional info about a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    desc_dict: dict of {str, int}, optional
        dictionary of correspondence between the operation base and
        the value in the content.
    parent: Message or None
        parent message.

    Notes
    -----
    The __getitem__ method was disallowed because only one word is expected.

    See Also
    --------
    FieldSingle: parent class.
    """

    READ = "r"
    WRITE = "w"
    ERROR = "e"

    def __init__(
            self,
            format_name: str,
            *,
            start_byte: int,
            fmt: str,
            desc_dict: dict[str, int] = None,
            info: dict[str, Any] | None = None,
            parent: Message = None,
    ):
        if desc_dict is None:
            self._desc_dict = {"r": 0, "w": 1, "e": 2}
        else:
            self._desc_dict = desc_dict
        self._desc_dict_r = {v: k for k, v in self._desc_dict.items()}

        SingleField.__init__(
            self,
            format_name,
            "operation",
            start_byte=start_byte,
            fmt=fmt,
            info=info,
            parent=parent
        )
        self._desc = ""

    def get_setter(self) -> FieldSetter:
        return FieldSetter.operation(
            fmt=self._fmt,
            desc_dict=self._desc_dict,
            info=self._info,
        )

    def update_desc(self) -> None:
        """Update desc by desc dict where key is a content value."""
        c_value = self.unpack()
        if len(c_value):
            if c_value[0] in self.desc_dict_r:
                self._desc = self._desc_dict_r[c_value[0]]
            else:
                self._desc = "unknown"
        else:
            self._desc = ""

    def compare(self, other) -> bool:
        """
        Compare operations between self and other.

        Parameters
        ----------
        other: str or OperationField or MessageType
            object for comparsion.

        Returns
        -------
        bool
            result of comparsion using == operator.

        Raises
        ------
        TypeError:
            if other is not instanse string, FieldOperation or MessageType.
        """
        if isinstance(other, str):
            base = other[0]
        elif isinstance(other, OperationField):
            base = other.base
        elif isinstance(other, MessageType):
            base = other.operation.base
        else:
            raise TypeError("invalid class for comparsion: %s" % type(other))

        return self.base == base

    def set(self, content: ContentType | str) -> None:
        if isinstance(content, str):
            c_value = self._desc_dict[content]
        else:
            c_value = content
        SingleField.set(self, c_value)
        self.update_desc()

    @property
    def base(self) -> str:
        """Operation base (fisrt letter of desc) or
        empty if content is empty."""
        return self._desc[0] if len(self._desc) else ""

    @property
    def desc(self) -> str:
        """Operation description. Can contain several letters."""
        return self._desc

    @property
    def desc_dict(self) -> dict[str, int]:
        """Dictionary of correspondence between the base content value."""
        return self._desc_dict

    @property
    def desc_dict_r(self) -> dict[int, str]:
        """Reversed desc_dict."""
        return self._desc_dict_r

    def __eq__(self, other) -> bool:
        """
        Compare operations between self and other.

        Parameters
        ----------
        other: str, FieldOperation or MessageType
            object for comparsion.

        Returns
        -------
        bool
            result of comparsion using == operator.

        See Also
        --------
        compare: comparsion method.
        """
        return self.compare(other)

    def __ne__(self, other) -> bool:
        """
        Compare operations between self and other.

        Parameters
        ----------
        other: str, FieldOperation or MessageType
            object for comparsion.

        Returns
        -------
        bool
            result of comparsion using != operator.

        See Also
        --------
        compare: comparsion method.
        """
        return not self.compare(other)


class ResponseField(SingleField):
    """
    Represents a field of a Message with response field.

    Parameters
    ----------
    format_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    info: dict of {str, Any}, optional
        additional info about a field.
    codes: dict of {int, Core or int}
        matching dictionary value and codes.
    default: int or Code or None
        default code if value undefined.
    parent: Message or None
        parent message.

    See Also
    --------
    SingleField: parent class.
    """

    OK = Code.OK
    "all is OK"

    WAIT = Code.WAIT
    "wait next message"

    RAISE = Code.RAISE
    "some error"

    UNDEFINED = Code.UNDEFINED
    "codes dict does not have value"

    def __init__(
            self,
            format_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            codes: dict[int, Code | int],
            default: int | Code | None = UNDEFINED,
            info: dict[str, Any] | None = None,
            parent: Message = None,
    ):
        super().__init__(
            format_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            info=info,
            may_be_empty=False,
            parent=parent,
        )

        self._codes = {}
        for k, v in codes.items():
            self._codes[k] = Code(v) if isinstance(v, int) else v
        if isinstance(default, int):
            default = Code(default)
        self._def_code = default

    def get_setter(self) -> FieldSetter:

        return FieldSetter.response(
            fmt=self._fmt,
            codes=self._codes,
            default=self._def_code,
            info=self._info,
        )

    @property
    def codes(self) -> dict[int, Code]:
        """
        Returns
        -------
        dict of {int, Code}
            matching dictionary value and codes
        """
        return self._codes

    @property
    def current_code(self) -> Code:
        """
        Returns
        -------
        Code
            current code.

        Raises
        ------
        FieldContentError
            if content is empty;
            if unknown value and default code is None.
        """
        if len(self):
            val = self.unpack()[0]
            if val in self._codes:
                return self._codes[val]
            elif self._def_code is not None:
                return self._def_code
            raise FieldContentError(
                self.__class__,
                clarification=f"undefined code by content {val}"
            )
        raise FieldContentError(
            self.__class__, clarification="content is empty"
        )

    @property
    def default_code(self) -> Code | None:
        """
        Returns
        -------
        Code or None
            default code or none.
        """
        return self._def_code

    def __eq__(self, other: Code | int) -> bool:
        """Compare current code with other"""
        return self.current_code == other

    def __ne__(self, other: Code | int) -> bool:
        """Compare current code with other"""
        return self.current_code != other


class FieldSetter(object):

    BYTES = DataLengthField.BYTES
    WORDS = DataLengthField.WORDS

    def __init__(
            self,
            special: str = None,
            **kwargs: Any,
    ):
        self.special = special
        if "default" in kwargs and isinstance(kwargs["default"], bytes):
            raise TypeError(
                "%s not recomended bytes type for 'default' argument" % type(
                    kwargs["default"]
                )
            )
        self.kwargs = kwargs

    @classmethod
    def base(
            cls,
            *,
            expected: int,
            fmt: str,
            default: ContentType = None,
            info: dict[str, Any] = None,
            may_be_empty: bool = False,
    ):
        """For classical field"""
        if default is None:
            default = []
        return cls(
            expected=expected,
            fmt=fmt,
            default=default,
            info=info,
            may_be_empty=may_be_empty
        )

    @classmethod
    def single(
            cls,
            *,
            fmt: str,
            default: ContentType = None,
            info: dict[str, Any] = None,
            may_be_empty: bool = False,
    ):
        if default is None:
            default = []
        return cls(
            special="single",
            fmt=fmt,
            default=default,
            info=info,
            may_be_empty=may_be_empty
        )

    @classmethod
    def static(
            cls,
            *,
            fmt: str,
            default: ContentType,
            info: dict[str, Any] = None,
    ):
        return cls(
            special="static",
            fmt=fmt,
            default=default,
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
    def crc(
            cls,
            *,
            fmt: str,
            info: dict[str, Any] | None = None,
            algorithm_name: str = "crc16-CCITT/XMODEM",
    ):
        return cls(
            special="crc", fmt=fmt, info=info, algorithm_name=algorithm_name
        )

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

    @classmethod
    def response(
            cls,
            *,
            fmt: str,
            codes: dict[int | float, Code | int],
            default: int | Code | None = Code.UNDEFINED,
            info: dict[str, Any] | None = None,
    ):
        return cls(
            special="response",
            fmt=fmt,
            codes=codes,
            default=default,
            info=info
        )

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        kwargs = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        if len(kwargs):
            kwargs = ", " + kwargs
        return f"<{cls_name}(special={self.special}{kwargs})>"

    __str__ = __repr__


FieldType = (
    Field |
    SingleField |
    StaticField |
    AddressField |
    CrcField |
    DataField |
    DataLengthField |
    OperationField |
    ResponseField
)


@runtime_checkable
class MessageType(Protocol):

    @property
    def address(self) -> AddressField:
        return AddressField("", start_byte=0, fmt="i")

    @property
    def data(self) -> DataField:
        return DataField(
            "",
            start_byte=0,
            expected=-1,
            fmt="i"
        )

    @property
    def data_length(self) -> DataLengthField:
        return DataLengthField("", start_byte=0, fmt="i")

    @property
    def operation(self):
        return OperationField("", start_byte=0, fmt="i")

    def __getitem__(self, field: str) -> FieldType: ...

    def __iter__(self) -> str: ...


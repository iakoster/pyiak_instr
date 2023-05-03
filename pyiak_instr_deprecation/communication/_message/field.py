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

from src.pyiak_instr.core import Code
from ...exceptions import (
    FieldContentError
)
if TYPE_CHECKING:
    from .message import FieldMessage


__all__ = [
    "ContentType",
    "Field",
    "FieldSetter",
    "AddressField",
    "CrcField",
    "DataField",
    "DataLengthField",
    "IdField",
    "OperationField",
    "SingleField",
    "StaticField",
    "ResponseField",
    "FieldType",
    "FieldContentError"
]


ContentType = (
        bytes | bytearray | int | float | Iterable |
        SupportsBytes | np.number | np.ndarray
)


class Field(BaseField):

    def get_setter(self) -> FieldSetter:
        """
        Get setter of the field instance.

        Returns
        -------
        FieldSetter
            setter for this field instance.
        """
        return FieldSetter.base(
            expected=self._exp,
            fmt=self._fmt,
            default=self.unpack_default(),
            may_be_empty=self._may_be_empty
        )

    def reset_to_default(self) -> None:
        """Set field content to default."""
        self._content = self._def

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

    def __str__(self) -> str:
        """
        Returns a string of hexadecimal numbers from the content.

        When converting, left insignificant zeros are removed and
        a space separator between words.

        Returns 'EMPTY' in field have no content.

        Returns
        -------
        str
            string representing the content in a readable format.
        """
        if not len(self._content):
            return "EMPTY"

        words = []
        for start in range(0, len(self.content), self.bytesize):
            word = self.content[start:start + self.bytesize] \
                .hex().lstrip("0")
            words.append(word.upper() if len(word) else "0")
        return " ".join(words)

    def __repr__(self) -> str:
        """Returns string representation of the field instance"""
        if self.words_count > 16:
            self_str = "{} ...({})".format(
                " ".join(str(self).split(" ")[:8]), self.words_count - 8
            )
        else:
            self_str = str(self)

        return f"<{self.__class__.__name__}({self_str}, fmt='{self._fmt}')>"


class SingleField(Field):
    """
    Represents a field of a Message with single word.

    Parameters
    ----------
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    may_be_empty: bool, default=False
        if True then field can be empty in a message.
    parent: FieldMessage or None
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
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            default: ContentType = b"",
            may_be_empty: bool = False,
            parent: FieldMessage = None,
    ):
        Field.__init__(
            self,
            mf_name,
            name,
            start_byte=start_byte,
            expected=1,
            fmt=fmt,
            default=default,
            may_be_empty=may_be_empty,
            parent=parent
        )

    def get_setter(self) -> FieldSetter:
        return FieldSetter.single(
            fmt=self._fmt,
            default=self.unpack_default(),
            may_be_empty=self._may_be_empty,
        )


class StaticField(SingleField):
    """
    Represents a field of a Message with static single word (e.g. preamble).

    Parameters
    ----------
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    default: Content, default=b""
        field content.
    parent: FieldMessage or None
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
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            default: ContentType,
            parent: FieldMessage = None,
    ):
        SingleField.__init__(
            self,
            mf_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            default=default,
            may_be_empty=False,
            parent=parent,
        )

    def get_setter(self) -> FieldSetter:
        return FieldSetter.static(
            fmt=self._fmt, default=self.unpack_default(),
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
            super().set(content)


class AddressField(SingleField):
    """
    Represents a field of a Message with address.

    Parameters
    ----------
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    may_be_empty: bool, default=False
        if True then field can be empty in a message.
    parent: FieldMessage or None
        parent message.

    See Also
    --------
    FieldSingle: parent class.
    """

    def __init__(
            self,
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            may_be_empty: bool = False,
            parent: FieldMessage = None,
    ):
        SingleField.__init__(
            self,
            mf_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            may_be_empty=may_be_empty,
            parent=parent
        )

    def get_setter(self) -> FieldSetter:
        return FieldSetter.address(fmt=self._fmt)


class CrcField(SingleField):
    """
    Represents a field of a Message with crc value.

    If the field name is 'crc', then Message will automatically update
    the value. The user/developer is responsible for setting the correct
    parameters (e.g. fmt).

    Parameters
    ----------
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of a field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    wo_fields: set[str], default=None
        list of the field names, which will not included for crc calculation.
    algorithm_name: str, default='crc16-CCITT/XMODEM'
        the name of the algorithm by which the crc is counted.
    parent: FieldMessage | None, default=None
        parent message.

    See Also
    --------
    FieldSingle: parent class.
    """

    def __init__(
            self,
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            algorithm_name: str = "crc16-CCITT/XMODEM",
            wo_fields: set[str] = None,
            parent: FieldMessage = None,
    ):
        if algorithm_name not in self.CRC_ALGORITHMS:
            raise ValueError("invalid algorithm name: %s" % algorithm_name)

        if wo_fields is None:
            wo_fields = set()

        SingleField.__init__(
            self,
            mf_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            parent=parent
        )
        self._wo_fields = wo_fields
        self._alg_name = algorithm_name
        self._alg = self.CRC_ALGORITHMS[algorithm_name]

    def calculate(self, msg: FieldMessage) -> int:
        """
        Calculate a crc value of a message with all fields except all
        crc fields.

        Parameters
        ----------
        msg: FieldMessage
            message.

        Returns
        -------
        int
            crc value.
        """
        content = b""
        for field in msg:
            if field.name not in self._wo_fields \
                    and not isinstance(field, CrcField):
                content += field.content
        return self._alg(content)

    def get_setter(self) -> FieldSetter:
        return FieldSetter.crc(
            fmt=self._fmt,
            algorithm_name=self._alg_name,
            wo_fields=self._wo_fields,
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

    @property
    def wo_fields(self) -> set[str]:
        """
        Returns
        -------
        set[str]
            list of the field names, which will not included for crc calculation.
        """
        return self._wo_fields


class DataField(Field):
    """
    Represents a field of a Message with data.

    Parameters
    ----------
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    parent: FieldMessage or None
        parent message.

    See Also
    --------
    Field: parent class.
    """

    def __init__(
            self,
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            expected: int,
            fmt: str,
            parent: FieldMessage = None
    ):
        Field.__init__(
            self,
            mf_name,
            name,
            start_byte=start_byte,
            expected=expected,
            fmt=fmt,
            may_be_empty=True,
            parent=parent
        )

    def append(self, content: ContentType) -> None:
        content = self._convert_content(content)
        if self._exp > 0:
            exp = self._exp + len(content) // self.bytesize
        else:
            exp = None
        self._content = self._validate_content(
            self._content + content, exp=exp
        )
        if self._exp > 0:
            self._exp = exp

    def get_setter(self) -> FieldSetter:
        return FieldSetter.data(expected=self._exp, fmt=self._fmt)


class DataLengthField(SingleField):
    """
    Represents a field of a Message with data length.

    Parameters
    ----------
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    may_be_empty: bool, default=False
        if True then field can be empty in a message.
    units: int
        data length units. Data can be measured in bytes or words.
    additive: int
        additional value to the length of the data.
    parent: FieldMessage or None
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

    def __init__(
            self,
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            may_be_empty: bool = False,
            behaviour: str = "actual",  # todo: logic
            units: Code | int = Code.BYTES,
            additive: int = 0,
            parent: FieldMessage = None
    ):
        if additive < 0 or not isinstance(additive, int):
            raise ValueError(
                "additive number must be positive integer, "
                f"got {additive}"
            )
        if behaviour not in {"actual", "expected2read"}:
            raise ValueError(
                "invalid behaviour: %r not in "
                "{'actual', 'expected2read}" % behaviour
            )

        if isinstance(units, int):
            units = Code(units)
        if units not in (Code.BYTES, Code.WORDS):
            raise ValueError("invalid units: %d" % units)

        SingleField.__init__(
            self,
            mf_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            may_be_empty=may_be_empty,
            parent=parent
        )
        self._bvr = behaviour
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

        if self._units is Code.BYTES:
            return len(data) + self._add
        elif self._units is Code.WORDS:
            return data.words_count + self._add
        else:
            raise ValueError(f"invalid units: {self._units}")

    def get_setter(self) -> FieldSetter:
        return FieldSetter.data_length(
            fmt=self._fmt,
            behaviour=self._bvr,
            units=self._units,
            additive=self._add,
        )

    def update(self) -> None:
        """
        Update data length content via parent message.

        Raises
        ------
        ValueError
            if parent is None;
            if units not in {BYTES, WORDS}.
        """
        if self._parent is None:
            raise ValueError("There is no parent message")
        self.set(self.calculate(self._parent.data))

    @property
    def additive(self) -> int:
        """Additional value to the data length."""
        return self._add

    @property
    def behaviour(self) -> str:
        """
        Returns
        -------
        str
            field behaviour.
        """
        return self._bvr

    @property
    def units(self) -> Code:
        """Data length units."""
        return self._units


class IdField(SingleField):
    """
    Represents a field with a unique identifier of a particular message.

    Parameters
    ----------
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    parent: FieldMessage or None
        parent message.
    """

    def __init__(
            self,
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            parent: FieldMessage = None,
    ):
        super().__init__(
            mf_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            parent=parent,
        )

    def get_setter(self) -> FieldSetter:
        return FieldSetter.id_field(fmt=self._fmt)

    def is_equal_to(self, other: IdField | int | float | bytes) -> bool:
        """
        Check that 'other' has the same value as the given instance.

        If 'other' is empty IdField - returns False.

        Parameters
        ----------
        other: IdField | int | float | bytes
            other 'IdField' instance or value.

        Returns
        -------
        bool
            True - values are equal, False - they are not.

        Raises
        ------
        TypeError
            if the type is impossible to compare.
        ValueError
            if self instance is empty.
        """
        if not len(self):
            raise ValueError("field is empty")

        match other:
            case IdField():
                if len(other):
                    return self._content == other.content
                return False
            case int() | float():
                return self[0] == other
            case bytes():
                return self._content == other
            case _:
                raise TypeError("invalid type: %s" % type(other))


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
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    may_be_empty: bool, default=False
        if True then field can be empty in a message.
    desc_dict: dict of {str, int}, optional
        dictionary of correspondence between the operation base and
        the value in the content.
    parent: FieldMessage or None
        parent message.

    See Also
    --------
    FieldSingle: parent class.
    """

    READ = "r"  # todo: to codes
    WRITE = "w"
    ERROR = "e"

    def __init__(
            self,
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            may_be_empty: bool = False,
            desc_dict: dict[str, int] = None,
            parent: FieldMessage = None,
    ):
        if desc_dict is None:
            self._desc_dict = {"r": 0, "w": 1, "e": 2}
        else:
            self._desc_dict = desc_dict
        self._desc_dict_r = {v: k for k, v in self._desc_dict.items()}

        SingleField.__init__(
            self,
            mf_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            may_be_empty=may_be_empty,
            parent=parent,
        )
        self._desc = ""

    def get_setter(self) -> FieldSetter:
        return FieldSetter.operation(
            fmt=self._fmt, desc_dict=self._desc_dict,
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
    mf_name: str
        the name of package format to which the field belongs.
    name: str
        the name of the field.
    start_byte: int
        the number of bytes in the message from which the fields begin.
    fmt: str
        format for packing or unpacking the content. The word length
        is calculated from the format.
    codes: dict of {int, Core or int}
        matching dictionary value and codes.
    default_code: int or Code or None
        default code if value undefined.
    parent: FieldMessage or None
        parent message.

    See Also
    --------
    SingleField: parent class.
    """

    OK = Code.OK
    "all is OK"

    WAIT = Code.WAIT
    "wait next message"

    ERROR = Code.ERROR
    "some error"

    UNDEFINED = Code.UNDEFINED
    "codes dict does not have value"

    def __init__(
            self,
            mf_name: str,
            name: str,
            *,
            start_byte: int,
            fmt: str,
            codes: dict[int, Code | int],
            default: ContentType = 0,
            default_code: int | Code | None = UNDEFINED,
            parent: FieldMessage = None,
    ):
        super().__init__(
            mf_name,
            name,
            start_byte=start_byte,
            fmt=fmt,
            default=default,
            may_be_empty=False,
            parent=parent,
        )

        self._codes = {}
        for k, v in codes.items():
            self._codes[k] = Code(v) if isinstance(v, int) else v
        if isinstance(default_code, int):
            default_code = Code(default_code)
        self._def_code = default_code

    def get_setter(self) -> FieldSetter:
        return FieldSetter.response(
            fmt=self._fmt, codes=self._codes, default_code=self._def_code,
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

    FIELD_TYPES = dict(
        single=SingleField,
        static=StaticField,
        address=AddressField,
        crc=CrcField,
        data=DataField,
        data_length=DataLengthField,
        id=IdField,
        operation=OperationField,
        response=ResponseField,
    )

    def __init__(
            self,
            field_type: str = None,
            **kwargs: Any,
    ):  # todo: test to default values
        self.field_type = field_type
        if "default" in kwargs and isinstance(kwargs["default"], bytes):
            raise TypeError(
                "%s not recommended bytes type for 'default' argument" % type(
                    kwargs["default"]
                )
            )
        self.kwargs = kwargs

    def get_field_class(self) -> type[FieldType]:  # todo: tests
        """
        Get field class by field type name.

        If `field_type` is None or another non-existent field type name -
        Field class is returned.

        Returns
        -------
        type[FieldType]
            field class instance.
        """
        return self.FIELD_TYPES.get(self.field_type, Field)

    @classmethod
    def base(
            cls,
            *,
            expected: int,
            fmt: str,
            default: ContentType = None,
            may_be_empty: bool = False,
    ):
        """For classical field"""
        if default is None:
            default = []
        return cls(
            expected=expected,
            fmt=fmt,
            default=default,
            may_be_empty=may_be_empty
        )

    @classmethod
    def single(
            cls,
            *,
            fmt: str,
            default: ContentType = None,
            may_be_empty: bool = False,
    ):
        if default is None:
            default = []
        return cls(
            field_type="single",
            fmt=fmt,
            default=default,
            may_be_empty=may_be_empty
        )

    @classmethod
    def static(cls, *, fmt: str, default: ContentType):
        return cls(field_type="static", fmt=fmt, default=default)

    @classmethod
    def address(cls, *, fmt: str, may_be_empty: bool = False):
        return cls(field_type="address", fmt=fmt, may_be_empty=may_be_empty)

    @classmethod
    def crc(
            cls,
            *,
            fmt: str,
            algorithm_name: str = "crc16-CCITT/XMODEM",
            wo_fields: set[str] | None = None
    ):
        return cls(
            field_type="crc",
            fmt=fmt,
            algorithm_name=algorithm_name,
            wo_fields=wo_fields,
        )

    @classmethod
    def data(cls, *, expected: int, fmt: str):
        return cls(field_type="data", expected=expected, fmt=fmt)

    @classmethod
    def data_length(
            cls,
            *,
            fmt: str,
            may_be_empty: bool = False,
            behaviour: str = "actual",
            units: int = Code.BYTES,
            additive: int = 0,
    ):
        return cls(
            field_type="data_length",
            fmt=fmt,
            may_be_empty=may_be_empty,
            behaviour=behaviour,
            units=units,
            additive=additive,
        )

    @classmethod
    def id_field(cls, *, fmt: str):
        return cls(field_type="id", fmt=fmt)

    @classmethod
    def operation(
            cls,
            *,
            fmt: str,
            desc_dict: dict[str, int] = None,
            may_be_empty: bool = False,
    ):
        return cls(
            field_type="operation",
            fmt=fmt,
            may_be_empty=may_be_empty,
            desc_dict=desc_dict,
        )

    @classmethod
    def response(
            cls,
            *,
            fmt: str,
            codes: dict[int | float, Code | int],
            default: ContentType = 0,
            default_code: int | Code | None = Code.UNDEFINED,
    ):
        return cls(
            field_type="response",
            fmt=fmt,
            codes=codes,
            default=default,
            default_code=default_code,
        )

    @property
    def init_kwargs(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict[str, Any]
            dictionary with all required arguments for init method.
        """
        return dict(
            field_type=self.field_type,
            **self.kwargs
        )

    def __eq__(self, other: Any) -> bool:
        """
        Compare this field to `other`.

        If `other` is not FieldSetter instance return False.

        Parameters
        ----------
        other: Any
            comparison object.

        Returns
        -------
        bool
            comparison result
        """
        if isinstance(other, FieldSetter):
            return self.field_type == other.field_type \
                   and self.kwargs == other.kwargs
        return False

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        kwargs = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        if len(kwargs):
            kwargs = ", " + kwargs
        return f"<{cls_name}(field_type={self.field_type}{kwargs})>"

    __str__ = __repr__


FieldType = (
    Field |
    SingleField |
    StaticField |
    AddressField |
    CrcField |
    DataField |
    DataLengthField |
    IdField |
    OperationField |
    ResponseField
)


@runtime_checkable
class MessageType(Protocol):

    @property
    def address(self) -> AddressField:
        return AddressField("", "", start_byte=0, fmt="i")

    @property
    def data(self) -> DataField:
        return DataField(
            "",
            "",
            start_byte=0,
            expected=-1,
            fmt="i"
        )

    @property
    def data_length(self) -> DataLengthField:
        return DataLengthField("", "", start_byte=0, fmt="i")

    @property
    def operation(self) -> OperationField:
        return OperationField("", "", start_byte=0, fmt="i")

    def __getitem__(self, field: str) -> FieldType: ...

    def __iter__(self) -> str: ...


"""Private module of ``pyiak_instr.communication.message`` with field
classes."""
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field as _field
from typing import Any, Callable, ClassVar, Self, Union

from ...core import Code
from ...store import BytesFieldParameters, BytesFieldPattern


__all__ = [
    "MessageFieldParameters",
    "SingleMessageFieldParameters",
    "StaticMessageFieldParameters",
    "AddressMessageFieldParameters",
    "CrcMessageFieldParameters",
    "DataMessageFieldParameters",
    "DataLengthMessageFieldParameters",
    "IdMessageFieldParameters",
    "OperationMessageFieldParameters",
    "ResponseMessageFieldParameters",
    "MessageFieldUnionT",
    "MessageFieldPattern",
]


@dataclass(frozen=True, kw_only=True)
class MessageFieldParameters(BytesFieldParameters):
    """Represents a general field of a Message."""


@dataclass(frozen=True, kw_only=True)
class SingleMessageFieldParameters(MessageFieldParameters):
    """
    Represents a field of a Message with single word.
    """

    expected: int = 1
    """expected number of words in the field. In Single fields always equal
    to one."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.expected != 1:
            raise ValueError("single field should expect one word")

    def validate(self, content: bytes) -> bool:
        """
        Check the content for compliance with the field parameters.

        Parameters
        ----------
        content: bytes
            content for validating.

        Returns
        -------
        bool
            True - content is correct, False - not.
        """
        if super().validate(content):
            return len(content) == self.bytes_expected
        return False


@dataclass(frozen=True, kw_only=True)
class StaticMessageFieldParameters(SingleMessageFieldParameters):
    """
    Represents a field of a Message with static single word (e.g. preamble).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.default) == 0:
            raise ValueError("default value not specified")

    def validate(self, content: bytes) -> bool:
        """
        Check the content for compliance with the field parameters.

        Parameters
        ----------
        content: bytes
            content for validating.

        Returns
        -------
        bool
            True - content is correct, False - not.
        """
        if super().validate(content):
            return content == self.default
        return False


@dataclass(frozen=True, kw_only=True)
class AddressMessageFieldParameters(SingleMessageFieldParameters):
    """
    Represents a field of a Message with address.
    """


@dataclass(frozen=True, kw_only=True)
class CrcMessageFieldParameters(SingleMessageFieldParameters):
    """
    Represents a field of a Message with crc value.
    """

    # todo: calculate CRC with values (e.g. poly) not full realization.

    algorithm_name: str = "crc16-CCITT/XMODEM"
    """the name of the algorithm by which the crc is counted."""

    wo_fields: set[str] = _field(default_factory=set)
    """list of the field names, which will not included for crc
    calculation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.algorithm_name not in self.CRC_ALGORITHMS:
            raise ValueError(
                f"invalid algorithm: {repr(self.algorithm_name)}"
            )

    @staticmethod
    def get_crc16_ccitt_xmodem(content: bytes) -> int:
        """
        Calculate a crc16-CCITT/XMODEM of content.

        Parameters
        ----------
        content : bytes
            content to calculate crc.

        Returns
        -------
        int
            crc value from 0 to 0xffff.
        """

        crc, poly = 0, 0x1021
        for byte in content:
            crc ^= byte << 8
            for _ in range(8):
                crc <<= 1
                if crc & 0x10000:
                    crc ^= poly
            crc &= 0xFFFF
        return crc

    CRC_ALGORITHMS: ClassVar[dict[str, Callable[[bytes], int]]] = {
        "crc16-CCITT/XMODEM": get_crc16_ccitt_xmodem
    }

    @property
    def algorithm(self) -> Callable[[bytes], int]:
        """
        Returns
        -------
        Callable[[bytes], int]
            algorithm to calculate crc.
        """
        return self.CRC_ALGORITHMS[self.algorithm_name]


@dataclass(frozen=True, kw_only=True)
class DataMessageFieldParameters(MessageFieldParameters):
    """Represents a field of a Message with data."""

    expected: int = 0


@dataclass(frozen=True, kw_only=True)
class DataLengthMessageFieldParameters(SingleMessageFieldParameters):
    """
    Represents a field of a Message with data length.
    """

    behaviour: Code = Code.ACTUAL  # todo: logic
    """determines the behavior of determining the content value."""

    units: Code = Code.BYTES
    """data length units. Data can be measured in bytes or words."""

    additive: int = 0
    """additional value to the length of the data."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.additive < 0:
            raise ValueError(
                "additive number must be positive integer, "
                f"got {self.additive}"
            )
        if self.behaviour not in {Code.ACTUAL}:
            raise ValueError(f"invalid behaviour: {repr(self.behaviour)}")
        if self.units not in {Code.BYTES, Code.WORDS}:
            raise ValueError(f"invalid units: {repr(self.units)}")


@dataclass(frozen=True, kw_only=True)
class IdMessageFieldParameters(SingleMessageFieldParameters):
    """
    Represents a field with a unique identifier of a particular message.
    """


@dataclass(frozen=True, kw_only=True)
class OperationMessageFieldParameters(SingleMessageFieldParameters):
    """
    Represents a field of a Message with operation (e.g. read).

    Operation codes are needed to compare the operation when receiving
    a message and generally to understand what operation is written in
    the message.

    If the dictionary is None, the standard value will be assigned
    {READ: 0, WRITE: 1}.
    """

    descriptions: dict[Code, int] = _field(
        default_factory=lambda: {Code.READ: 0, Code.WRITE: 1}
    )
    """dictionary of correspondence between the operation base and the value
    in the content."""

    _desc_r: dict[int, Code] = _field(default_factory=dict, repr=False)
    """reversed `descriptions`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "_desc_r", {v: k for k, v in self.descriptions.items()}
        )


@dataclass(frozen=True, kw_only=True)
class ResponseMessageFieldParameters(SingleMessageFieldParameters):
    """
    Represents a field of a Message with response field.
    """

    codes: dict[int, Code] = _field(default_factory=dict)
    """matching dictionary value and codes."""

    default_code: Code = Code.UNDEFINED
    """default code if value undefined."""


MessageFieldUnionT = Union[
    MessageFieldParameters,
    SingleMessageFieldParameters,
    StaticMessageFieldParameters,
    AddressMessageFieldParameters,
    CrcMessageFieldParameters,
    DataMessageFieldParameters,
    DataLengthMessageFieldParameters,
    IdMessageFieldParameters,
    OperationMessageFieldParameters,
    ResponseMessageFieldParameters,
]


# todo: typehint - Generic. .get, .get_updated and ._get_field_class writes
#  just because needed to change type.
class MessageFieldPattern(BytesFieldPattern):
    """
    Represents class which storage common parameters for message field.

    Parameters
    ----------
    field_type: str
        name of field type.
    **parameters: Any
        common parameters.
    """

    _FIELD_TYPES = dict(
        basic=MessageFieldParameters,
        single=SingleMessageFieldParameters,
        static=StaticMessageFieldParameters,
        address=AddressMessageFieldParameters,
        crc=CrcMessageFieldParameters,
        data=DataMessageFieldParameters,
        data_length=DataLengthMessageFieldParameters,
        id=IdMessageFieldParameters,
        operation=OperationMessageFieldParameters,
        response=ResponseMessageFieldParameters,
    )
    """dictionary of field types where key is a name of type."""

    def __init__(self, field_type: str, **parameters: Any) -> None:
        if field_type not in self._FIELD_TYPES:
            raise ValueError(f"invalid field type name: {field_type}")
        super().__init__(field_type=field_type, **parameters)

    def get(self, **parameters: Any) -> MessageFieldUnionT:
        """
        Get field initialized with parameters from pattern and from
        `parameters`.

        Parameters
        ----------
        **parameters: Any
            additional field initialization parameters.

        Returns
        -------
        MessageFieldUnionT
            initialized field.
        """
        return self._get_field_class()(**self._kw, **parameters)

    def get_updated(self, **parameters: Any) -> MessageFieldUnionT:
        """
        Get field initialized with parameters from pattern and from
        `parameters`.

        If parameters from pattern will be updated via `parameters` before
        creation Field instance.

        Parameters
        ----------
        **parameters: Any
            parameters for field.

        Returns
        -------
        MessageFieldUnionT
            initialized field.
        """
        kw_ = self._kw.copy()
        kw_.update(parameters)
        return self._get_field_class()(**kw_)

    def _get_field_class(self) -> type[MessageFieldUnionT]:
        """
        Get field class by `field_type`.

        Returns
        -------
        type[MessageFieldUnionT]
            message field class.
        """
        return self._FIELD_TYPES.get(  # type: ignore[return-value]
            self._field_type, MessageFieldParameters
        )

    @classmethod
    def basic(
        cls,
        fmt: Code,
        expected: int,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> Self:
        """
        Get pattern for BasicMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        expected : int
            expected number of words in the field.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.
        default : bytes, default=b''
            default value of the field.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="basic",
            fmt=fmt,
            expected=expected,
            order=order,
            default=default,
        )

    @classmethod
    def single(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",
    ) -> Self:
        """
        Get pattern for SingleMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.
        default : bytes, default=b''
            default value of the field.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="single",
            fmt=fmt,
            order=order,
            default=default,
        )

    @classmethod
    def static(
        cls,
        fmt: Code,
        default: bytes,
        order: Code = Code.BIG_ENDIAN,
    ) -> Self:
        """
        Get pattern for StaticMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        default : bytes
            default value of the field.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="static",
            fmt=fmt,
            default=default,
            order=order,
        )

    @classmethod
    def address(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
    ) -> Self:
        """
        Get pattern for AddressMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="address",
            fmt=fmt,
            order=order,
        )

    @classmethod
    def crc(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        algorithm_name: str = "crc16-CCITT/XMODEM",
        wo_fields: set[str] | None = None,
    ) -> Self:
        """
        Get pattern for CrcMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.
        algorithm_name : str, default='crc16-CCITT/XMODEM'
            the name of the algorithm.
        wo_fields: set[str] | None, default=None
            list of the field names, which will be ignored for crc
            calculation.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        if wo_fields is None:
            wo_fields = set()
        return cls(
            field_type="crc",
            fmt=fmt,
            order=order,
            algorithm_name=algorithm_name,
            wo_fields=wo_fields,
        )

    @classmethod
    def data(
        cls,
        fmt: Code,
        expected: int = 0,
        order: Code = Code.BIG_ENDIAN,
    ) -> Self:
        """
        Get pattern for DataMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        expected : int, default=0
            expected number of words in the field.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="data",
            fmt=fmt,
            expected=expected,
            order=order,
        )

    # pylint: disable=too-many-arguments
    @classmethod
    def data_length(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        behaviour: Code = Code.ACTUAL,
        units: Code = Code.BYTES,
        additive: int = 0,
    ) -> Self:
        """
        Get pattern for DataLengthMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.
        behaviour : Code, default=Code.ACTUAL
            determines the behavior of determining the content value.
        units : Code, default=Code.BYTES
            data length units. Data can be measured in bytes or words.
        additive : int, default=0
            additional value to the length of the data.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="data_length",
            fmt=fmt,
            order=order,
            behaviour=behaviour,
            units=units,
            additive=additive,
        )

    # pylint: disable=invalid-name
    @classmethod
    def id(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
    ) -> Self:
        """
        Get pattern for IdMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="id",
            fmt=fmt,
            order=order,
        )

    @classmethod
    def operation(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        descriptions: dict[Code, int] | None = None,
    ) -> Self:
        """
        Get pattern for OperationMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.
        descriptions: dict[Code, int] | None, default=None
            dictionary of correspondence between the operation base and the
            value in the content.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        if descriptions is None:
            descriptions = {Code.READ: 0, Code.WRITE: 1}
        return cls(
            field_type="operation",
            fmt=fmt,
            order=order,
            descriptions=descriptions,
        )

    # pylint: disable=too-many-arguments
    @classmethod
    def response(
        cls,
        fmt: Code,
        order: Code = Code.BIG_ENDIAN,
        codes: dict[int, Code] | None = None,
        default_code: Code = Code.UNDEFINED,
        default: bytes = b"",
    ) -> Self:
        """
        Get pattern for ResponseMessageField.

        Parameters
        ----------
        fmt : Code
            format code.
        order : Code, default=Code.BIG_ENDIAN
            bytes order code.
        codes : dict[int, Code] | None, default=None
            matching dictionary value and codes.
        default_code: Code = Code.UNDEFINED
            default code if value undefined.
        default: bytes, default=b''
            default value of the field.

        Returns
        -------
        Self
            initialized class with defined parameters.
        """
        return cls(
            field_type="response",
            fmt=fmt,
            order=order,
            codes=codes,
            default_code=default_code,
            default=default,
        )

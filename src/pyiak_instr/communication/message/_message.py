"""Private module of ``pyiak_instr.communication.message`` with field
parsers."""
import numpy.typing as npt

from ._field import (
    MessageField,
    SingleMessageField,
    StaticMessageField,
    AddressMessageField,
    CrcMessageField,
    DataMessageField,
    DataLengthMessageField,
    IdMessageField,
    OperationMessageField,
    ResponseMessageField,
)
from ...store import BytesFieldParser
from ...core import Code

__all__ = [
    "MessageFieldParser",
    "SingleMessageFieldParser",
    "StaticMessageFieldParser",
    "AddressMessageFieldParser",
    "CrcMessageFieldParser",
    "DataMessageFieldParser",
    "DataLengthMessageFieldParser",
    "IdMessageFieldParser",
    "OperationMessageFieldParser",
    "ResponseMessageFieldParser",
]


# todo: all parsers via generic
class MessageFieldParser(BytesFieldParser):
    """
    Represents parser for work with message field content.
    """

    _f: MessageField

    @property
    def fld(self) -> MessageField:
        """
        Returns
        -------
        MessageField
            field instance
        """
        return self._f


class SingleMessageFieldParser(MessageFieldParser):
    """
    Represents parser for work with single message field content.
    """

    _f: SingleMessageField

    @property
    def fld(self) -> SingleMessageField:
        """
        Returns
        -------
        SingleMessageField
            field instance
        """
        return self._f


class StaticMessageFieldParser(MessageFieldParser):
    """
    Represents parser for work with static message field content.
    """

    _f: StaticMessageField

    @property
    def fld(self) -> StaticMessageField:
        """
        Returns
        -------
        StaticMessageField
            field instance.
        """
        return self._f


class AddressMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with crc message field content.
    """

    _f: AddressMessageField

    @property
    def fld(self) -> AddressMessageField:
        """
        Returns
        -------
        AddressMessageField
            field instance.
        """
        return self._f


class CrcMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with crc message field content.
    """

    _f: CrcMessageField

    def calculate(self) -> int:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        # content = b""
        # for field in msg:
        #     if field.name not in self.fld.wo_fields and field is not self:
        #         content += field.content
        # return self.fld.algorithm(content)
        raise NotImplementedError()

    def update(self) -> None:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> CrcMessageField:
        """
        Returns
        -------
        CrcMessageField
            field instance.
        """
        return self._f


class DataMessageFieldParser(MessageField):
    """
    Represents parser for work with data message field content.
    """

    _f: DataMessageField

    def append(self, content: npt.ArrayLike) -> None:
        """
        append new content.

        Parameters
        ----------
        content : ArrayLike
            new content.

        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> DataMessageField:
        """
        Returns
        -------
        DataMessageField
            field instance.
        """
        return self._f


class DataLengthMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with data length message field content.
    """

    _f: DataLengthMessageField

    def calculate(self) -> None:
        """
        Raises
        -------
        NotImplementedError
            placeholder
        """
        raise NotImplementedError()

    def update(self) -> None:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> DataLengthMessageField:
        """
        Returns
        -------
        DataLengthMessageField
            field instance.
        """
        return self._f


class IdMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with id message field content.
    """

    _f: IdMessageField

    def compare(self) -> None:
        """
        Returns
        -------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> IdMessageField:
        """
        Returns
        -------
        IdMessageField
            field instance.
        """
        return self._f


class OperationMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with operation message field content.
    """

    _f: OperationMessageField

    def calculate(self) -> None:
        """
        Raises
        -------
        NotImplementedError
            placeholder
        """
        raise NotImplementedError()

    def update(self) -> None:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> OperationMessageField:
        """
        Returns
        -------
        OperationMessageField
            self instance.
        """
        return self._f


class ResponseMessageFieldParser(SingleMessageFieldParser):
    """
    Represents parser for work with response message field content.
    """

    _f: ResponseMessageField

    @property
    def code(self) -> Code:
        """
        Raises
        ------
        NotImplementedError
            placeholder.
        """
        raise NotImplementedError()

    @property
    def fld(self) -> ResponseMessageField:
        """
        Returns
        -------
        ResponseMessageField
            self instance.
        """
        return self._f

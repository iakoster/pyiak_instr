from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any

from tinydb.table import Table, Document

from .field import FieldSetter
from .message import Message
from ...utilities import StringEncoder
from ...rwfile import (
    RWNoSqlJsonDatabase,
)


__all__ = [
    "AsymmetricResponseField",
    "MessageFormat",
]


class AsymmetricResponseField(object):  # todo: add to fields
    """
    Represents class for detecting error code in an incoming message.

    This class is supposed to be used when the asymmetric message format
    is on the other side of the PC and this asymmetric field corresponds
    to the status.

    If `AsymmetricResponseField.is_empty` then there is no any checks.
    Class is considered empty if the operand is an empty string.

    Field will be cut from incoming message in range [start, stop).

    Parameters
    ----------
    operand: {'==', '!='}, default=''
        How to behave when check .

        * ''(empty string): this class is empty.
        * ==: field is error when it is equal to value.
        * !=: field is error when it is equal to value.
    value: bytes
        value for checking. Can be coded via StringEncoder to string.
    start: int
        start byte of the error mark in a message.
    stop: int
        stop byte (not included) of the field in a message.

    Raises
    ------
    ValueError
        if `operand` in not valid.
        if `start` and `stop` is None.
    TypeError
        if `value` cannot be converted from string to bytes by StringEncoder.
        if `value` type is not bytes.
    """

    def __init__(
            self,
            operand: str = "",
            value: str | bytes = None,
            start: int = None,
            stop: int = None,
    ):
        if not len(operand):
            self._oper, self._val, self._slice = "", b"", slice(None)
            return

        if operand not in {"==", "!="}:
            raise ValueError(f"invalid operand: %r" % operand)
        if start is None or stop is None:
            raise ValueError("start or stop is not specified")
        if stop <= start:
            raise ValueError("stop <= start")
        self._oper, self._slice = operand, slice(start, stop)

        if isinstance(value, str):
            value = StringEncoder.from_str(value)
            if not isinstance(value, bytes):
                raise TypeError(
                    "value can't be converted from string to bytes"
                )
        elif not isinstance(value, bytes):
            raise TypeError("invalid type of value")
        self._val = value

    def match(self, msg: bytes) -> tuple[bytes | Message, bool]:
        """
        Match field in a message.

        The second return value is a boolean value and indicates the presence
        of an error in the message (if True).

        If `operand` is None (class is empty) returns message as is and
        False indicator.

        Parameters
        ----------
        msg: bytes
            message for check.

        Returns
        -------
        tuple of {bytes or Message, bool}
            the edited message (if part of it is cut out) and
            the error mark indicator.

        Raises
        ------
        TypeError
            if `msg` type is not bytes.

        """
        if self.is_empty:
            return msg, False
        if not isinstance(msg, bytes):
            raise TypeError("bytes message required")

        return (
            msg[:self.start] + msg[self.stop:],
            self._validate_field(msg[self._slice])
        )

    def _validate_field(self, field: bytes) -> bool:
        """
        Compare field with value by operand.

        Parameters
        ----------
        field: bytes
            asymmetric response field.

        Returns
        -------
        bool
            comparison result
        """
        match self._oper:
            case "==":
                return self._val == field
            case "!=":
                return self._val != field
        assert False, f"invalid operand: {self._oper}"

    @property
    def is_empty(self) -> bool:
        """
        Returns
        -------
        bool
            indicator that there is an empty error mark.
        """
        return not len(self._oper)

    @property
    def kwargs(self) -> dict[str, str | int | bytes | None]:
        """
        Bytes type will be encoded to string.

        Returns
        -------
        dict[str, str | int | bytes | None]
            dictionary for writing to a package format file.
        """
        if self.is_empty:
            return {}

        return dict(
            operand=self._oper,
            value=self._val,
            start=self.start,
            stop=self.stop
        )

    @property
    def operand(self) -> str | None:
        """
        Returns
        -------
        str
            match operation.
        """
        return self._oper

    @property
    def start(self) -> int | None:
        """
        Returns
        -------
        int
            the number of a start byte of an error mark.
        """
        return self._slice.start

    @property
    def stop(self) -> int | None:
        """
        Returns
        -------
        int
            the number of a stop byte of an error mark.
        """
        return self._slice.stop

    @property
    def value(self) -> bytes:
        """
        Returns
        -------
        bytes of list of int or float
            value for checking.
        """
        return self._val


class MessageFormat(object):
    """
    Represents class instance for message format.

    Parameters
    ----------
    emark: AsymmetricResponseField
        error mark for message.
    **settings: FieldSetter or Any
        settings for message. If there is a FieldSetter that it will be added
        to setters dict and to msg_args in other cases.

    Raises
    ------
    ValueError
        if not all required fields are specified.
    """

    def __init__(
            self,
            emark: AsymmetricResponseField = AsymmetricResponseField(),
            **settings: FieldSetter | Any
    ):
        self._emark = emark
        self._msg_args = {}
        self._setters = {}
        for k, v in settings.items():
            if isinstance(v, FieldSetter):
                self._setters[k] = v
            else:
                self._msg_args[k] = v

        setters_diff = set(self._setters) - set(Message.REQ_FIELDS)
        if len(setters_diff):
            ValueError(
                f"not all requared setters were got: %s are missing" %
                ", ".join(setters_diff)
            )

    def write(self, format_table: Table) -> None:
        """
        Write parameters to the table.

        Parameters
        ----------
        format_table: Table
            table instance.
        """

        def drop_none(dict_: dict[Any]) -> Any:
            return {k: v for k, v in dict_.items() if v is not None}

        def remove_if_doc_id_exists(doc_ids: list[int]) -> None:
            for doc_id in doc_ids:
                if format_table.contains(doc_id=doc_id):
                    format_table.remove(doc_ids=(doc_id,))

        remove_if_doc_id_exists([-2, -1])
        format_table.insert(Document(drop_none(self._msg_args), doc_id=-1))
        format_table.insert(Document(self._emark.kwargs, doc_id=-2))

        for i_setter, (name, setter) in enumerate(self.setters.items()):
            field_pars = {"name": name}
            if setter.special is not None:
                field_pars["special"] = setter.special
            field_pars.update(drop_none(setter.kwargs))
            remove_if_doc_id_exists([i_setter])
            format_table.insert(Document(field_pars, doc_id=i_setter))

    def get(self, **update: dict[str, Any]) -> Message:
        """
        Get message instance with message format.

        Parameters
        ----------
        update: dict[str, Any]
            dictinary of parameters to change.

        Returns
        -------
        Message
            message configured with message format.
        """
        setters = deepcopy(self._setters)
        if len(update):
            for setter_name, fields in update.items():
                setters[setter_name].kwargs.update(fields)
        return Message(**self._msg_args).configure(**setters)

    @classmethod
    def read(cls, path: Path, fmt_name: str) -> MessageFormat:
        """
        Read message format from a json database.

        Parameters
        ----------
        path: Path
            path to json database.
        fmt_name: str
            name of format.

        Returns
        -------
        MessageFormat
            message format initilized with parameters from database.
        """

        with RWNoSqlJsonDatabase(path) as db:
            if fmt_name not in db.tables():
                raise ValueError(
                    "The format not exists in the database: %s" % fmt_name
                )

            table = db.table(fmt_name)
            msg = dict(table.get(doc_id=-1))

            setters = {}
            for field_id in range(len(table) - 1):
                field = dict(table.get(doc_id=field_id))
                setters[field.pop("name")] = FieldSetter(**field)

        return cls(**msg, **setters)

    @property
    def emark(self) -> AsymmetricResponseField:
        """
        Returns
        -------
        AsymmetricResponseField
            error mark.
        """
        return self._emark

    @property
    def msg_args(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict[str, Any]
            arguments for setting Message class.
        """
        return self._msg_args

    @property
    def setters(self) -> dict[str, FieldSetter]:
        """
        Returns
        -------
        dict[str, FieldSetter]
            setters for Message.configure method.
        """
        return self._setters

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from tinydb.table import Table, Document

from .message import FieldSetter, Message
from .register import Register, RegisterMap
from pyinstr_iakoster.utilities import StringEncoder
from pyinstr_iakoster.rwfile import (
    RWNoSqlJsonDatabase,
    RWSQLite,
)


__all__ = [
    "MessageErrorMark",
    "MessageFormat",
    "PackageFormat",
]


class MessageErrorMark(object):  # todo: make only for asymmetric emark (bytes only)
    """
    A class for detecting the existence of an error mark in a message.

    Attribute `operation` must be one of {'eq', 'neq'}.
        - 'eq' -- equal. If equal to value that error mark in a message.
        - 'neq' -- not equal. It not equal to value that error mark
            in a message.

    Parameters
    ----------
    operation: str
        match operation. If None, instance is considered empty
        (all checks pass).
    value: str or bytes or list of int or float
        value for checking. Can be encoded with StringEncoder.
    start_byte: int
        start byte of the error mark in a message. Used for cutting
        an error mark from a message in bytes.
    stop_byte: int
        stop byte (not included) of the error mark in a message. Used
        for cutting an error mark from a message in bytes.
    field_name: str
        the name of the field in Message instance. Used for copying
        an error mark from an initilized message.

    Raises
    ------
    ValueError
        if `operation` not in {'eq', 'neq'};
        if `field_name` not specified and one of `start_byte` or `stop_byte`
            not specified too;
        if `field_name`, `start_byte` and `stop_byte` is specified;
        if `value` is string and cannot be converted from string
            by StringEncoder.
    TypeError
        if `start_byte` and `stop_byte` are specified and `value` not instance
        of bytes type;
        if `value` type is not bytes or list.

    See Also
    --------
    MessageErrorMark.match: main method of this class.
    """

    def __init__(
            self,
            operation: str = None,
            value: str | bytes | list[int | float] = b"",
            start_byte: int = None,
            stop_byte: int = None,
            field_name: str = None,
    ):
        self._oper = operation
        if self.empty:
            self._field_name, self._slice, self._val = None, slice(None), b""
            return

        if operation not in (None, "eq", "neq"):
            raise ValueError("%r operation not in {'eq', 'neq'}" % operation)
        if field_name is None and None in (start_byte, stop_byte):
            raise ValueError(
                "field name or start and stop bytes must be defined"
            )
        if None not in (field_name, start_byte, stop_byte):
            raise ValueError(
                "field_name, start_byte, stop_byte cannot be specified at "
                "the same time"
            )
        self._field_name = field_name
        self._slice = slice(start_byte, stop_byte)

        if isinstance(value, str):
            value = StringEncoder.from_str(value)
            if isinstance(value, str):
                raise ValueError("convert string is impossible")
        if self.bytes_required and not isinstance(value, bytes):
            raise TypeError("invalid type: %s, expected bytes" % type(value))
        if not isinstance(value, bytes | list):
            raise TypeError("invalid type: %s" % type(value))
        self._val = value

    def exists(self, msg: bytes | Message) -> tuple[bytes | Message, bool]:
        """
        Match error mark in a message.

        The second return value is a boolean value and indicates the presence
        of an error in the message (if True).

        If `operation` is None (error mark is empty) returns message as is and
        False indicator.

        If `start_byte` and `stop_byte` are specified, the message is expected
        as bytes. This part of the message is cut from the message.

        If `field_name` is specified, the message is expected as bytes.

        Parameters
        ----------
        msg: bytes or Message
            message for checking.

        Raises
        ------
        TypeError
            if `start_byte` and `stop_byte` are specified and message
            not is instance of bytes type.

        Returns
        -------
        tuple of {bytes or Message, bool}
            the edited message (if part of it is cut out) and
            the error mark indicator.
        """
        if self._oper is None:
            return msg, False
        if self._field_name is None and not isinstance(msg, bytes):
            raise TypeError("bytes type required")

        if isinstance(msg, bytes):
            msg_mark = msg[self._slice]
            msg = msg[:self._slice.start] + msg[self._slice.stop:]

        else:
            msg_mark = msg[self._field_name]
            if isinstance(self._val, bytes):
                msg_mark = msg_mark.content
            else:
                msg_mark = msg_mark.unpack().tolist()

        if self._oper == "eq":
            return msg, msg_mark == self._val # if mark == val, then mark is error mark
        elif self._oper == "neq":
            return msg, msg_mark != self._val # if mark != val, then mark is error mark

    @property
    def bytes_required(self) -> bool:
        """
        Returns
        -------
        bool
            indicator that '.match' method expected message in bytes.
        """
        return self._field_name is None

    @property
    def empty(self) -> bool:
        """
        Returns
        -------
        bool
            indicator that there is an empty error mark.
        """
        return self._oper is None

    @property
    def field_name(self) -> str | None:
        """
        Returns
        -------
        str
            the name of the field.
        """
        return self._field_name

    @property
    def kwargs(self) -> dict[str, str | int | bytes | None]:
        """
        Bytes type will be encoded to string.

        Returns
        -------
        dict[str, str | int | bytes | None]
            dictionary for writing to a package format file.
        """
        if self._oper is None:
            return {}

        ret: dict[str, str | list[int | float] | bytes | None] = {
            "operation": self._oper
        }
        if isinstance(self._val, bytes):
            ret["value"] = StringEncoder.to_str(self._val)
        else:
            ret["value"] = self._val
        if self.bytes_required:
            ret["start_byte"] = self._slice.start
            ret["stop_byte"] = self._slice.stop
        else:
            ret["field_name"] = self._field_name

        return ret

    @property
    def operation(self) -> str | None:
        """
        Returns
        -------
        str
            match operation.
        """
        return self._oper

    @property
    def start_byte(self) -> int | None:
        """
        Returns
        -------
        int
            the number of a start byte of an error mark.
        """
        return self._slice.start

    @property
    def stop_byte(self) -> int | None:
        """
        Returns
        -------
        int
            the number of a stop byte of an error mark.
        """
        return self._slice.stop

    @property
    def value(self) -> bytes | list[int | float]:
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
    emark: MessageErrorMark
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
            emark: MessageErrorMark = MessageErrorMark(),
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
    def emark(self) -> MessageErrorMark:
        """
        Returns
        -------
        MessageErrorMark
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


class PackageFormat(object):
    """
    Represents class instance for package format.

    Parameters
    ----------
    **formats: MessageFormat
        message formats. Key is a name of a message format.
    """

    def __init__(
            self,
            register_map: RegisterMap = None,
            **formats: MessageFormat
    ):
        if register_map is None:
            register_map = RegisterMap(
                pd.DataFrame(columns=RegisterMap.EXPECTED_COLUMNS)
            )

        self._formats = formats
        for name, mf in self._formats.items():
            self._formats[name].msg_args["mf_name"] = name
        self._reg_map = register_map

    def write(
            self,
            message_format: Path,
            register_map: Path = None
    ) -> None:
        """
        Write parameters to the table.

        The database will be cleared before writing the data.

        Parameters
        ----------
        message_format: Path
            save path for message_format database.
        register_map: Path
            save path for register map database.
        """
        with RWNoSqlJsonDatabase(message_format) as db:
            db.hapi.drop_tables()
            for name, format_ in self._formats.items():
                format_.write(db[name])

        if register_map is not None:
            with RWSQLite(register_map) as db:
                for table in db.tables:
                    db.request(f"DROP TABLE {table};")
                self._reg_map.write(db.connection)

    def get(self, mf_name: str, **update: dict[str, Any]) -> Message:
        """
        Get message instance with message format.

        Parameters
        ----------
        mf_name: str
            the name of the message format.
        **update: dict[str, Any]
            dictinary of parameters to change.

        Returns
        -------
        Message
            message configured with selected message format.
        """
        return self._formats[mf_name].get(**update)

    def get_format(self, format_name: str) -> MessageFormat:
        """
        Get message format by name.

        Parameters
        ----------
        format_name: str
            message format name.

        Returns
        -------
        MessageFormat
            selected message format.
        """
        return self._formats[format_name]

    def get_register(self, register: str) -> Register:
        """
        Get register by its name or extended name.

        Parameters
        ----------
        register: str
            register name.

        Returns
        -------
        Register
            register instance.
        """
        return self._reg_map[register, self]

    def read_register_map(self, database: Path) -> PackageFormat:
        """
        Read register map from database.

        Parameters
        ----------
        database: Path
            path to the database.

        Returns
        -------
        PackageFormat
            self instance.
        """
        self._reg_map = RegisterMap.read(database)
        return self

    def set_register_map(self, reg_map: RegisterMap) -> PackageFormat:
        """
        Set register map.

        Parameters
        ----------
        reg_map: RegisterMap
            register map instance.

        Returns
        -------
        PackageFormat
            self instance.
        """
        self._reg_map = reg_map
        return self

    @classmethod
    def read(cls, database: Path) -> PackageFormat:
        """
        Read all message formats from a json database.

        Parameters
        ----------
        database: Path
            path to json database.

        Returns
        -------
        PackageFormat
            package format initilized by database.
        """
        formats = {}
        with RWNoSqlJsonDatabase(database) as db:
            for table_name in db.hapi.tables():
                table = db[table_name]

                msg_args = table.get(doc_id=-1)
                if table.contains(doc_id=-2):
                    emark = MessageErrorMark(**table.get(doc_id=-2))
                else:
                    emark = MessageErrorMark()

                setters = {}
                for i_setter in range(len(table)):
                    if not table.contains(doc_id=i_setter):
                        break
                    setter_args = table.get(doc_id=i_setter)
                    name = setter_args.pop("name")
                    setters[name] = FieldSetter(**setter_args)
                formats[table_name] = MessageFormat(emark=emark, **msg_args, **setters)
        return cls(**formats)

    @property
    def formats(self) -> dict[str, MessageFormat]:
        """
        Returns
        -------
        dict[str, MessageFormat]
            all existing message formats in the package format.
        """
        return self._formats

    @property
    def register_map(self) -> RegisterMap:
        """
        Returns
        -------
        RegisterMap
            register map instance.
        """
        return self._reg_map

    def __getattr__(self, register: str) -> Register:
        """
        Get register.

        If not exists, exception will be raised.

        Parameters
        ----------
        register: str
            register name.

        Returns
        -------
        Register
            register instance.
        """
        return self.get_register(register)

    def __getitem__(self, register: str) -> Register:
        """
        Get register.

        If not exists, exception will be raised.

        Parameters
        ----------
        register: str
            register name.

        Returns
        -------
        Register
            register instance.
        """
        return self.get_register(register)

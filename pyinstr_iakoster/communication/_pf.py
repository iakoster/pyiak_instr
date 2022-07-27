from copy import deepcopy
from pathlib import Path
from typing import Any

from tinydb.table import Document

from ._msg import FieldSetter, Message
from ..utilities import StringEncoder
from ..rwfile import (
    RWNoSqlJsonDatabase,
)


__all__ = [
    "MessageErrorMark",
    "MessageFormat",
    "PackageFormat",
]


class MessageErrorMark(object): # nodesc

    def __init__(
            self,
            operation: str = None,
            value: str | bytes | list[int | float] = None,
            start_byte: int = None,
            stop_byte: int = None,
            field_name: str = None,
    ):
        if operation is None:
            self._empty = True
            self._oper, self._field_name = "", ""
            self._start, self._stop = 0, 0
            self._val = b""
            self._bytes_req = True
            return
        else:
            self._empty = False

        if operation not in ("eq", "neq"):
            raise ValueError("%r operation not in {'eq', 'neq'}" % operation)
        if field_name is None and None in (start_byte, stop_byte):
            raise ValueError(
                "field name or start byte and end byte must be defined"
            )
        if value is None:
            raise ValueError("value not specified")

        self._oper = operation
        if field_name is not None:
            self._field_name = field_name
            self._start, self._stop = 0, 0
            self._bytes_req = False

        else:
            self._field_name = ""
            self._start, self._stop = start_byte, stop_byte
            self._bytes_req = True

        if isinstance(value, str):
            value = StringEncoder.from_str(value)
            if isinstance(value, str):
                raise ValueError("convert string is impossible")

        if self._bytes_req and not isinstance(value, bytes):
            raise TypeError(
                "if start and end bytes is specified that value must be bytes"
            )
        self._val = value

    def match(self, msg: bytes | Message) -> tuple[bytes | Message, bool]: # nodesc
        if self._empty:
            return msg, False
        if self._bytes_req and not isinstance(msg, bytes):
            raise TypeError("bytes type required")

        if isinstance(msg, bytes):
            msg_mark = msg[self._start:self._stop]
            msg = msg[:self._start] + msg[self._stop:]

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
        return self._bytes_req

    @property
    def empty(self) -> bool:
        return self._empty

    @property
    def field_name(self) -> str:
        return self._field_name

    @property
    def kwargs(self):
        if self._empty:
            return {}

        ret = {"operation": self._oper}
        if isinstance(self._val, bytes):
            ret["value"] = StringEncoder.to_str(self._val)
        else:
            ret["value"] = self._val
        if self._bytes_req:
            ret["start_byte"] = self._start
            ret["stop_byte"] = self._stop
        else:
            ret["field_name"] = self._field_name

        return ret

    @property
    def operation(self) -> str:
        return self._oper

    @property
    def start_byte(self) -> int:
        return self._start

    @property
    def stop_byte(self) -> int:
        return self._stop

    @property
    def value(self) -> bytes | list[int | float]:
        return self._val


class MessageFormat(object):  # nodesc

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

    def write(self, format_table: RWNoSqlJsonDatabase.table_class) -> None:

        def drop_none(dict_: dict[Any]) -> Any:
            new_dict = {}
            for k, v in dict_.items():
                if v is not None:
                    new_dict[k] = v
            return new_dict # todo dict comprehetion

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
        setters = deepcopy(self._setters)
        if len(update):
            for setter_name, fields in update.items():
                setters[setter_name].kwargs.update(fields)
        return Message(**self._msg_args).configure(**setters)

    @classmethod
    def read(cls, path: Path, fmt_name: str):

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
        return self._emark

    @property
    def msg_args(self) -> dict[str, Any]:
        return self._msg_args

    @property
    def setters(self) -> dict[str, FieldSetter]:
        return self._setters


class PackageFormat(object):  # nodesc

    def __init__(
            self,
            **formats: MessageFormat
    ):
        self._formats = formats
        for name, mf in self._formats.items():
            self._formats[name].msg_args["format_name"] = name

    def write(self, database: Path) -> None:
        with RWNoSqlJsonDatabase(database) as db:
            db.drop_tables()
            for name, format_ in self._formats.items():
                format_.write(db.table(name))

    def get(self, format_name: str, **update: dict[str, Any]) -> Message:
        return self[format_name].get(**update)

    @classmethod
    def read(cls, database: Path):
        formats = {}
        with RWNoSqlJsonDatabase(database) as db:
            for table_name in db.tables():
                table = db.table(table_name)

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
        return self._formats

    def __getitem__(self, format_name: str) -> MessageFormat:
        return self._formats[format_name]

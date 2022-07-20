from pathlib import Path
from typing import Any

from tinydb.table import Document

from ._msg import FieldSetter, Message
from ..rwfile import (
    RWNoSqlJsonDatabase,
)


__all__ = [
    "MessageFormat"
]


class MessageFormat(object):

    def __init__(
            self,
            **settings: FieldSetter | Any
    ):

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

        format_table.insert(Document(drop_none(self._msg_args), doc_id=-1))

        for i_setter, (name, setter) in enumerate(self.setters.items()):
            field_pars = {"name": name}
            if setter.special is not None:
                field_pars["special"] = setter.special
            field_pars.update(drop_none(setter.kwargs))
            format_table.insert(Document(field_pars, doc_id=i_setter))

    def get(self) -> Message:
        return Message(**self._msg_args).set(**self._setters)

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
    def msg_args(self) -> dict[str, Any]:
        return self._msg_args

    @property
    def setters(self) -> dict[str, FieldSetter]:
        return self._setters


class PackageFormat(object):

    def __init__(
            self,
            **formats: MessageFormat
    ):
        self._formats = formats

    def write(self, database: Path) -> None:
        with RWNoSqlJsonDatabase(database) as db:
            db.truncate()
            for name, format_ in self._formats.items():
                format_.write(db.table(name))

    @classmethod
    def read(cls, database: Path):
        formats = {}
        with RWNoSqlJsonDatabase(database) as db:
            print(db.tables())
        return cls(**formats)

    def get(self, format_name: str) -> Message:
        return self[format_name].get()

    @property
    def formats(self) -> dict[str, MessageFormat]:
        return self._formats

    def __getitem__(self, format_name: str) -> MessageFormat:
        return self._formats[format_name]

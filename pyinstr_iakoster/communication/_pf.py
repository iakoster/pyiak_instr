from copy import deepcopy
from pathlib import Path
from typing import Any

from tinydb.table import Document

from ._msg import FieldSetter, Message
from ..rwfile import (
    RWNoSqlJsonDatabase,
)


__all__ = [
    "MessageFormat",
    "PackageFormat",
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

        def remove_if_doc_id_exists(doc_id: int) -> None:
            if format_table.contains(doc_id=doc_id):
                format_table.remove(doc_ids=(doc_id,))

        remove_if_doc_id_exists(-1)
        format_table.insert(Document(drop_none(self._msg_args), doc_id=-1))

        for i_setter, (name, setter) in enumerate(self.setters.items()):
            field_pars = {"name": name}
            if setter.special is not None:
                field_pars["special"] = setter.special
            field_pars.update(drop_none(setter.kwargs))
            remove_if_doc_id_exists(i_setter)
            format_table.insert(Document(field_pars, doc_id=i_setter))

    def get(self, **update: dict[str, Any]) -> Message:
        setters = deepcopy(self._setters)
        if len(update):
            for setter_name, fields in update.items():
                setters[setter_name].kwargs.update(fields)
        return Message(**self._msg_args).set(**setters)

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
                setters = {}
                for i_setter in range(len(table) - 1):
                    setter_args = table.get(doc_id=i_setter)
                    name = setter_args.pop("name")
                    setters[name] = FieldSetter(**setter_args)
                formats[table_name] = MessageFormat(**msg_args, **setters)
        return cls(**formats)

    @property
    def formats(self) -> dict[str, MessageFormat]:
        return self._formats

    def __getitem__(self, format_name: str) -> MessageFormat:
        return self._formats[format_name]

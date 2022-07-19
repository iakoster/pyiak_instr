import inspect
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
        self._message = {}
        self._setters = {}
        for k, v in settings.items():
            if isinstance(v, FieldSetter):
                self._setters[k] = v
            else:
                self._message[k] = v

        setters_diff = set(self._setters) - set(Message.REQ_FIELDS)
        if len(setters_diff):
            ValueError(
                f"not all requared setters were got: %s are missing" %
                ", ".join(setters_diff)
            )

    def write_pf(self, path: Path) -> None:

        def drop_none(dict_: dict[Any]) -> Any:
            new_dict = {}
            for k, v in dict_.items():
                if v is not None:
                    new_dict[k] = v
            return new_dict

        with RWNoSqlJsonDatabase(path) as db:
            table = db.table(self._message["format_name"])
            table.truncate()
            table.insert(Document(drop_none(self._message), doc_id=-1))

            for i_setter, (name, setter) in enumerate(self.setters.items()):
                field_pars = {"name": name}
                if setter.special is not None:
                    field_pars["special"] = setter.special
                field_pars.update(drop_none(setter.kwargs))
                table.insert(Document(field_pars, doc_id=i_setter))

    @classmethod
    def read_pf(cls, path: Path, fmt_name: str):

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
                name = field.pop("name")
                special = field.pop("special") if "special" in field else None
                setters[name] = FieldSetter(special=special, **field)

        return cls(**msg, **setters)

    @property
    def message(self) -> dict[str, Any]:
        return self._message

    @property
    def setters(self) -> dict[str, FieldSetter]:
        return self._setters

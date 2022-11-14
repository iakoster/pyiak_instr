from __future__ import annotations
from pathlib import Path
from typing import Any

import pandas as pd

from .field import FieldSetter
from .message import Message
from .register import Register, RegisterMap
from .message_format import MessageFormat, MessageErrorMark
from ...rwfile import (
    RWNoSqlJsonDatabase,
    RWSQLite,
)


__all__ = [
    "PackageFormat",
]


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

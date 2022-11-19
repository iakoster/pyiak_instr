from __future__ import annotations
from pathlib import Path
from typing import Any

from .message import Message
from .register import Register, RegisterMap
from .message_format import (
    MessageFormat,
    MessageFormatMap,
)


__all__ = [
    "PackageFormat",
]


class PackageFormat(object):
    """
    Represents class instance for package format.

    Parameters
    ----------
    registers: RegisterMap, default=RegisterMap()
        registers map instance.
    formats: MessageFormatMap, default=MessageFormatMap()
        message formats map instance.
    """

    def __init__(
            self,
            registers: RegisterMap = RegisterMap(),
            formats: MessageFormatMap = MessageFormatMap(),
    ):
        self._reg_map = registers
        self._mf_map = formats

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
        return self._mf_map[mf_name].get(**update)

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
        return self._mf_map[format_name]

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

    def write(
            self, *, formats: Path = None, registers: Path = None
    ) -> None:
        """
        Write parameters to the table.

        The database will be cleared before writing the data.

        Parameters
        ----------
        formats: Path, default=None
            save path for message formats config.
        registers: Path, default=None
            save path for register map database.
        """
        if formats is not None:
            self._mf_map.write(formats)
        if registers is not None:
            self._reg_map.write(registers)

    @classmethod
    def read(
            cls, *, formats: Path = None, registers: Path = None
    ) -> PackageFormat:
        """
        Read all message formats from a json database.

        Parameters
        ----------
        formats: Path, default=None
            path to message formats config.
        registers: Path, default=None
            path to registers database.

        Returns
        -------
        PackageFormat
            package format instance.
        """
        kw = {}
        if formats is not None:
            kw["formats"] = MessageFormatMap.read(formats)
        if registers is not None:
            kw["registers"] = RegisterMap.read(registers)
        return cls(**kw)

    @property
    def message_format_map(self) -> MessageFormatMap:
        """
        Returns
        -------
        dict[str, MessageFormat]
            all existing message formats in the package format.
        """
        return self._mf_map

    @property
    def register_map(self) -> RegisterMap:
        """
        Returns
        -------
        RegisterMap
            register map instance.
        """
        return self._reg_map

    def __getitem__(self, register: str) -> Register:
        """
        Get register by name.

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

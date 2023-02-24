from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any

from .field import FieldSetter
from .message import MessageType, MessageSetter
from ...rwfile import (
    RWConfig,
)


__all__ = [
    "AsymmetricResponseField",
    "MessageFormat",
    "MessageFormatMap",
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

        if not isinstance(value, bytes):
            raise TypeError("invalid type of value")
        self._val = value

    def match(self, msg: bytes) -> tuple[bytes, bool]:
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

    def __eq__(self, other: Any) -> bool:
        """
        Compare this field to `other`.

        If `other` is not AsymmetricResponseField instance return False.

        Parameters
        ----------
        other: Any
            comparison object.

        Returns
        -------
        bool
            comparison result
        """
        if isinstance(other, AsymmetricResponseField):
            return self.kwargs == other.kwargs
        return False


class MessageFormat(object):
    """
    Represents class instance for message format.

    Parameters
    ----------
    message_setter: MessageSetter
        message setter for specified format.
    arf: AsymmetricResponseField
        asymmetric field for message or kwargs for it.
    **setters: FieldSetter
        setters for message.

    Raises
    ------
    ValueError
        if not all required fields are specified.
    """

    def __init__(
            self,
            message_setter: dict[str, Any] | MessageSetter = MessageSetter(),
            arf: dict[str, Any] | AsymmetricResponseField = AsymmetricResponseField(),
            **setters: FieldSetter | dict[str, Any]
    ):
        if isinstance(arf, dict):
            arf = AsymmetricResponseField(**arf)
        if isinstance(message_setter, dict):
            message_setter = MessageSetter(**message_setter)
        for name, setter in setters.items():
            if isinstance(setter, dict):
                setters[name] = FieldSetter(**setter)

        self._arf = arf
        self._msg_set = message_setter
        self._setters = setters

        setters_diff = set(
            self._msg_set.message_class.REQUIRED_FIELDS
        ) - set(self._setters)
        if len(setters_diff):
            raise ValueError(f"missing the required setters: {setters_diff}")

    def write(self, config: Path) -> None:
        """
        Write parameters to the config.

        Parameters
        ----------
        config: Path
            path to the config file.
        """
        with RWConfig(config) as rwc:
            if self._msg_set.mf_name in rwc.hapi.sections():
                rwc.hapi.remove_section(self._msg_set.mf_name)
            rwc.apply_changes()  # todo: test to correct work (replace section)
            rwc.write(self._msg_set.mf_name, self.init_kwargs)

    def get(self, **update: dict[str, Any]) -> MessageType:
        """
        Get message instance with message format.

        Keywords arguments must be in format FIELD={PARAMETER: VALUE}.

        Parameters
        ----------
        update: dict[str, Any]
            dictionary of parameters to change.

        Returns
        -------
        MessageType
            message configured with message format.
        """
        setters = deepcopy(self._setters)
        if len(update):
            for setter_name, fields in update.items():
                setters[setter_name].kwargs.update(fields)
        return self._msg_set.message.configure(**setters)

    @classmethod
    def read(cls, config: Path, mf_name: str) -> MessageFormat:
        """
        Read message format from a config.

        Parameters
        ----------
        config: Path
            path to json database.
        mf_name: str
            name of message format.

        Returns
        -------
        MessageFormat
            message format initialized with parameters from database.

        Raises
        ------
        ValueError
            if config does now have required message format.
        """
        kw = {}
        with RWConfig(config) as rwc:
            if mf_name not in rwc.hapi.sections():
                raise ValueError("format with name %r not exists" % mf_name)
            kw = {opt: rwc.get(mf_name, opt)
                  for opt in rwc.hapi.options(mf_name)}
        return cls(**kw)

    @property
    def arf(self) -> AsymmetricResponseField:
        """
        Returns
        -------
        AsymmetricResponseField
            error mark.
        """
        return self._arf

    @property
    def init_kwargs(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict[str, Any]
            dictionary with all required arguments for init method.
        """
        return dict(
            message_setter=self._msg_set.init_kwargs,
            arf=self.arf.kwargs,
            **{n: s.init_kwargs for n, s in self._setters.items()}
        )

    @property
    def message_setter(self) -> MessageSetter:
        """
        Returns
        -------
        MessageSetter
            message setter instance.
        """
        return self._msg_set

    @property
    def setters(self) -> dict[str, FieldSetter]:
        """
        Returns
        -------
        dict[str, FieldSetter]
            setters for Message.configure method.
        """
        return self._setters


class MessageFormatMap(object):
    """
    Represents class with Message formats.

    Parameters
    ----------
    *formats: MessageFormat
        message formats where key is a message format name.
    """

    def __init__(self, *formats: MessageFormat):
        self._formats = {
            mf.message_setter.kwargs["mf_name"]: mf for mf in formats
        }

    def get(self, mf_name: str) -> MessageFormat:
        """
        Get message format by name.

        Parameters
        ----------
        mf_name: str
            message format name.

        Returns
        -------
        MessageFormat
            message format instance.

        Raises
        ------
        ValueError
            if name not in formats list.
        """
        if mf_name not in self._formats:
            raise ValueError("there is no format with name %r" % mf_name)
        return self._formats[mf_name]

    def write(self, config: Path) -> None:
        """
        Write formats to the config file.

        Parameters
        ----------
        config: Path
            path to config file.
        """
        for mf in self._formats.values():
            mf.write(config)

    @classmethod
    def read(cls, config: Path) -> MessageFormatMap:
        """
        Read message formats from config.
        
        Parameters
        ----------
        config: Path
            path to the config file.

        Returns
        -------
        MessageFormatMap
            class instance with formats from config.
        """
        with RWConfig(config) as rwc:
            formats = rwc.hapi.sections()
        return cls(*(MessageFormat.read(config, f) for f in formats))

    @property
    def formats(self) -> dict[str, MessageFormat]:
        """
        Returns
        -------
        dict[str, MessageFormat]
            dictionary of message formats, where key if format name.
        """
        return self._formats

    def __getitem__(self, mf_name: str) -> MessageFormat:
        """
        Get message format by name.

        Parameters
        ----------
        mf_name: str
            the name of a MessageFormat.

        Returns
        -------
        MessageFormat
            message format instance.
        """
        return self.get(mf_name)

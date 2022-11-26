from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any

from .field import FieldSetter
from .message import FieldMessage
from ...core import Code
from ...utilities import StringEncoder
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

        if isinstance(value, str):
            value = StringEncoder.from_str(value)
            if not isinstance(value, bytes):
                raise TypeError(
                    "value can't be converted from string to bytes"
                )
        elif not isinstance(value, bytes):
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
    arf: AsymmetricResponseField
        asymmetric field for message or kwargs for it.
    mf_name: str, default='std'
        name of the message format.
    splittable: bool, default=True
        shows that the message can be divided by the data.
    slice_length: int, default=1024
        max length of the data in one slice.
    **setters: FieldSetter
        setters for message.

    Raises
    ------
    ValueError
        if not all required fields are specified.
    """

    SEP = "__"
    "separator for nested dictionaries"

    def __init__(
            self,
            arf: dict[str, Any] | AsymmetricResponseField = AsymmetricResponseField(),
            mf_name: str = "std",
            splittable: bool = False,
            slice_length: int = 1024,
            **setters: FieldSetter
    ):
        if isinstance(arf, dict):
            arf = AsymmetricResponseField(**arf)
        self._arf = arf
        self._message = dict(
            mf_name=mf_name,
            splittable=splittable,
            slice_length=slice_length
        )
        self._setters = setters

        setters_diff = set(FieldMessage.REQUIRED_FIELDS) - set(self._setters)
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

        mf_name, mf_dict = self._message["mf_name"], {}
        sec_message = f"{mf_name}{self.SEP}message"
        sec_setters = f"{mf_name}{self.SEP}setters"

        mf_dict[sec_message] = {}
        for opt, val in dict(arf=self.arf.kwargs, **self._message).items():
            mf_dict[sec_message][opt] = StringEncoder.to_str(val)

        mf_dict[sec_setters] = {}
        for opt, val in self._setters.items():

            kw = val.kwargs  # fixme: fix krutch
            for k, v in kw.items():
                if isinstance(v, Code):
                    kw[k] = v.value
                elif isinstance(v, dict):
                    for k_, v_ in v.items():
                        if isinstance(v_, Code):
                            v[k_] = v_.value

            mf_dict[sec_setters][opt] = StringEncoder.to_str(
                dict(field_type=val.field_type, **val.kwargs)
            )

        with RWConfig(config) as rwc:

            for sec in rwc.hapi.sections():
                if sec.split(self.SEP)[0] == mf_name:
                    rwc.hapi.remove_section(sec)
            rwc.apply_changes()
            rwc.write(mf_dict)

    def get(self, **update: dict[str, Any]) -> FieldMessage:
        """
        Get message instance with message format.

        Keywords arguments must be in format FIELD={PARAMETER: VALUE}.

        Parameters
        ----------
        update: dict[str, Any]
            dictionary of parameters to change.

        Returns
        -------
        FieldMessage
            message configured with message format.
        """
        setters = deepcopy(self._setters)
        if len(update):
            for setter_name, fields in update.items():
                setters[setter_name].kwargs.update(fields)
        return FieldMessage(**self._message).configure(**setters)

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

        kw, setters = {}, {}
        sec_message = f"{mf_name}{cls.SEP}message"
        sec_setters = f"{mf_name}{cls.SEP}setters"

        with RWConfig(config) as rwc:
            sections = rwc.hapi.sections()
            if sec_message not in sections or sec_setters not in sections:
                raise ValueError("format with name %r not exists" % mf_name)

            for sec in [sec_message, sec_setters]:
                for opt in rwc.hapi.options(sec):
                    val = StringEncoder.from_str(
                        rwc.get(sec, opt, convert=False)
                    )  # todo: StringEncoder to RWConfig
                    if isinstance(val, str) and val[0] == StringEncoder.SOH:
                        val = StringEncoder.from_str(val + "\t")

                    if sec == sec_message:
                        kw[opt] = val
                    elif sec == sec_setters:
                        setters[opt] = FieldSetter(**val)

        return cls(**kw, **setters)

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
    def message(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict[str, Any]
            arguments for setting Message class.
        """
        return self._message

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
        self._formats = {mf.message["mf_name"]: mf for mf in formats}

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
        with RWConfig(config) as rwc:
            rwc.write({"master": {
                "formats": StringEncoder.to_str(list(self._formats))
            }})

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
            formats = StringEncoder.from_str(rwc.get("master", "formats", convert=False))
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

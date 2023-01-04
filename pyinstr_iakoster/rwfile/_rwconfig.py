import io
import configparser
from pathlib import Path
from typing import overload, Any

from ._core import RWFile
from ..utilities import StringEncoder


__all__ = ['RWConfig']


class RWConfig(RWFile):
    """
    Class for reading and writing to the configfile as *.ini.

    Include autoconverter for values.

    Parameters
    ----------
    filepath: Path or path-like str
        path to config file *.ini.
    """

    _hapi: configparser.ConfigParser

    FILE_SUFFIXES = {".ini"}

    def __init__(self, filepath: Path | str):
        super().__init__(filepath)
        self._hapi = self._read_config()

    def apply_changes(self) -> None:
        """
        Write configparser to the configfile.

        Used for save changes which created by .set method.
        """
        with io.open(self._fp, 'w') as cfg_file:
            self._hapi.write(cfg_file)

    def close(self):
        pass

    def get(self, section: str, option: str, convert: bool = True) -> Any:
        """
        Get value from the configparser.

        If convert is True, then the value will try to convert to
        some type (e.g. to the tuple with ints, list with bools, etc.)
        by ._str2any method.
        If convert is False the value will be returned 'as is'
        as a string.

        Parameters
        ----------
        section: str
            section name.
        option: str
            option name.
        convert: bool
            convert the resulting value from str.

        Returns
        -------
        Any
            resulting value from configfile.
        """
        value = self._hapi.get(section, option)
        if convert:
            return StringEncoder.from_str(value)
        return value

    def read(self, section: str, option: str, convert: bool = True) -> Any:
        """
        Read value from the configfile.

        Parameters
        ----------
        section: str
            section name.
        option: str
            option name.
        convert: bool
            convert the resulting value from str.

        Returns
        -------
        Any
            value from the configfile.
        """

        value = self._read_config().get(section, option)
        if convert:
            return StringEncoder.from_str(value)
        return value

    def set(
            self, section: str, option: str, value: Any, convert: bool = True
    ) -> None:
        """
        Set the value to the configparser.

        Does not change value in the configfile.
        Converts value to string by ._any2str method.

        Parameters
        ----------
        section: str
            section name.
        option: str
            option name.
        value: Any
            value to be set.
        convert: bool
            convert the resulting value to str by StringEncoder.

        See Also
        --------
        _any2str: method to convert the value to a str.
        """
        if convert:
            value = StringEncoder.to_str(value)
        self._hapi.set(section, option, value)

    def update_config(self) -> None:
        """Re-read the configfile from specified and writes to the class."""
        self._hapi = self._read_config()

    @overload
    def write(
            self, section: str, option: str, value: Any, convert: bool = True
    ) -> None:
        """
        Parameters
        ----------
        section: str
            section name.
        option: str
            option name.
        value: Any
            value for writing.
        convert: bool
            convert the resulting value to str by StringEncoder.
        """
        ...

    @overload
    def write(
            self, dictionary: dict[str, dict[str, Any]], convert: bool = True
    ) -> None:
        """
        Parameters
        ----------
        dictionary: dict of {str: {str: Any}}
            dictionary of values in format {section: {option: value}}.
        convert: bool
            convert the resulting value to str by StringEncoder.
        """
        ...

    def write(self, *args, convert: bool = True) -> None:
        """
        write(section: str, option: str, value: Any) -> None.
        write(dictionary: dict) -> None.

        Write value or dict to the configfile.

        Also writes value or dict to the configparser.

        Parameters
        ----------
        *args
            arguments for sets value to section, option
        convert: bool
            convert the resulting value to str by StringEncoder.
        """

        cfg = self._read_config()

        match args:
            case (str() as sec, str() as opt, val):
                if convert:
                    val = StringEncoder.to_str(val)
                cfg.set(sec, opt, val)
                self.set(sec, opt, val)

            case (dict() as dictionary,):
                vals = {}
                for sec, item in dictionary.items():
                    if sec not in vals:
                        vals[sec] = {}
                    for opt, val in item.items():
                        if convert:
                            val = StringEncoder.to_str(val)
                        vals[sec][opt] = val
                cfg.read_dict(vals)
                self._hapi.read_dict(vals)

            case _:
                raise TypeError(f"invalid arguments {args}")

        with io.open(self._fp, 'w') as cnfg_file:
            cfg.write(cnfg_file)

    def _read_config(self) -> configparser.ConfigParser:
        """
        Read config from filepath.

        If the config on the specified path does not exist,
        creates an empty config file.

        Returns
        -------
        configparser.ConfigParser
            config contains settings from file path.
        """

        config = configparser.ConfigParser()
        if self._fp.exists():
            config.read(self._fp)
        else:
            with io.open(self._fp, 'w') as cfg_file:
                config.write(cfg_file)
        return config

    @property
    def hapi(self) -> configparser.ConfigParser:
        return self._hapi

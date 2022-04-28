import io
import re
import configparser
from pathlib import Path
from typing import overload, Any

from ._utils import *


__all__ = ['RWConfig']


class RWConfig(object):
    """
    Class for reading and writing to the configfile as *.ini.

    Include autoconverter for values.

    Parameters
    ----------
    filepath: Path or path-like str
        path to config file *.ini.
    """

    LIST_DELIMITER = ','
    LIST_PATTERN = re.compile(LIST_DELIMITER)
    TUPLE_DELIMITER = ';'
    TUPLE_PATTERN = re.compile(TUPLE_DELIMITER)

    INT_PATTERN = re.compile('^\d+$')
    FLOAT_PATTERN = re.compile('^\d+\.\d+$')
    EFLOAT_PATTERN = re.compile('^\d\.\d+[eE][+-]\d+$')

    FILENAME_PATTERN = re.compile('\S+.ini$')

    def __init__(self, filepath: Path | str):
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath
        self._cfg = self.read_config()
        self.update_config()

    def update_config(self) -> None:
        """
        Re-reads the config file from specified and
        writes to the class.
        """
        self._cfg = self.read_config()

    def read_config(self) -> configparser.ConfigParser:
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
        if self._filepath.exists():
            config.read(self._filepath)
        else:
            with io.open(self._filepath, 'w') as cfg_file:
                config.write(cfg_file)
        return config

    def set(self, section: str, option: str, value: Any) -> None:
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

        See Also
        --------
        _any2str: method to convert the value to a str.
        """
        self._cfg.set(section, option, self._any2str(value))

    def apply_changes(self) -> None:
        """
        Write configparser to the configfile.

        Used for save changes which created by .set method.
        """
        with io.open(self._filepath, 'w') as cfg_file:
            self._cfg.write(cfg_file)

    def get(self, section: str, option: str,
            convert: bool = True) -> Any:
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
        value = self._cfg.get(section, option)
        return self._str2any(value) if convert else value

    @overload
    def write(self, section: str, option: str, value: Any) -> None:
        """
        Parameters
        ----------
        section: str
            section name.
        option: str
            option name.
        value: Any
            value to be write.
        """
        ...

    @overload
    def write(self, dictionary: dict) -> None:
        """
        Parameters
        ----------
        dictionary: dict of {str: {str: Any}}
            dictionary of values in format {section: {option: value}}.
        """
        ...

    def write(self, *args) -> None:
        """
        write(section: str, option: str, value: Qny) -> None.
        write(dictionary: dict) -> None.

        Write value or dict to the configfile.

        Also writes value or dict to the configparser.

        Parameters
        ----------
        *args
            arguments for sets value to section, option
        """

        match args:
            case (str() as section, str() as option, value):
                conv_value = self._any2str(value)
                config = configparser.ConfigParser()
                config.read(self._filepath)
                config.set(section, option, conv_value)
                self.set(section, option, conv_value)
                with io.open(self._filepath, 'w') as cnfg_file:
                    config.write(cnfg_file)

            case (dict() as dictionary,):
                conv_dict = {}
                for section, item in dictionary.items():
                    conv_dict[section] = {}
                    for option, raw_value in item.items():
                        conv_dict[section][option] = self._any2str(raw_value)
                self._cfg.read_dict(conv_dict)
                with io.open(self._filepath, 'w') as cnfg_file:
                    self._cfg.write(cnfg_file)

            case _:
                raise TypeError('Wrong args for write method')

    def _any2str(self, value) -> str:
        """
        Convert any value to the string.

        There is a specific rules for converting
        dict, tuple and list.

        Parameters
        ----------
        value: Any
            value for converting.

        Returns
        -------
        str
            value as str
        """

        def val2str(val: int | float | str) -> str:
            return str(val)

        if isinstance(value, dict):
            return self.LIST_DELIMITER.join(
                [self.TUPLE_DELIMITER.join(val2str(i) for i in item)
                 for item in value.items()])

        elif isinstance(value, tuple):
            return self.TUPLE_DELIMITER.join(
                val2str(v) for v in value)

        elif isinstance(value, list):
            return self.LIST_DELIMITER.join(
                val2str(v) for v in value)

        else:
            return val2str(value)

    def read(self, section: str, option: str) -> Any:
        """
        Read value from the configfile.

        Parameters
        ----------
        section: str
            section name.
        option: str
            option name.

        Returns
        -------
        Any
            value from the configfile.
        """

        config = configparser.ConfigParser()
        config.read(self._filepath)
        return self._str2any(config.get(section, option))

    def _str2any(self, value: str) -> Any:
        """
        Convert string value to the any.

        If there are no templates to convert, then returns
        the value 'as is' as a string.

        Parameters
        ----------
        value: str
            string value.

        Returns
        -------
        Any
            Converted value.
        """

        def val2any(val: str) -> int | float | str | bool:
            if self.INT_PATTERN.match(val) is not None:
                return int(val)

            elif (self.FLOAT_PATTERN.match(val) or
                  self.EFLOAT_PATTERN.match(val)) is not None:
                return float(val)

            elif val in ('True', 'False', 'None'):
                return eval(val)

            else:
                return val

        if None not in (self.TUPLE_PATTERN.search(value),
                        self.LIST_PATTERN.search(value)):
            raw_dict = [item.split(self.TUPLE_DELIMITER)
                        for item in value.split(self.LIST_DELIMITER)]
            return {val2any(raw_key): val2any(raw_val)
                    for (raw_key, raw_val) in raw_dict}

        elif self.TUPLE_PATTERN.search(value) is not None:
            return tuple(val2any(v) for v in
                         value.split(self.TUPLE_DELIMITER))

        elif self.LIST_PATTERN.search(value) is not None:
            return [val2any(v) for v in
                    value.split(self.LIST_DELIMITER)]

        else:
            return val2any(value)

    @property
    def config(self):
        """
        Returns
        -------
        configparser.ConfigParser
        """
        return self._cfg

    @property
    def filepath(self):
        """
        Returns
        -------
        Path
            path to the configfile
        """
        return self._filepath

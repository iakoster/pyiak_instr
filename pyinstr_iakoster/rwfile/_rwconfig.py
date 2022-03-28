import io
import re
import configparser
from pathlib import Path
from typing import overload, Any


from ._rwf_utils import *


__all__ = ['RWConfig']


class RWConfig(object):
    """
    Class for reading and writing to the configfile as *.ini.
    Include autoconverter for values.
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
        """
        :param filepath: path to .ini configfile
        """
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath
        self._cfg = self.read_config()
        self.update_config()

    def update_config(self) -> None:
        """
        Re-reads the config file and writes to the class

        :return: None
        """
        self._cfg = self.read_config()

    def read_config(self) -> configparser.ConfigParser:
        """
        Read config from filepath

        :return: readed configfile as ConfigParser
        :rtype: configparser.ConfigParser
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
        set the value for the configparser

        DOES NOT CHANGE the configfile

        :param section: section name string
        :param option: option name string
        :param value: value to be set in the section option
        :return: None
        """
        self._cfg.set(section, option, self._any2str(value))

    def apply_changes(self) -> None:
        """
        write configparser to the configfile

        Used for save changes which created by .set method

        :return: None
        """
        with io.open(self._filepath, 'w') as cfg_file:
            self._cfg.write(cfg_file)

    def get(self, section: str, option: str,
            convert: bool = True) -> Any:
        """
        get value from the configparser

        Convert is a boolean value.
        If it is False the value will be returned 'as is'
        as a string.
        If it is True, then the value will try to convert to
        some type (e.g. to the tuple with ints, list with bools, etc.)

        :param section: section name string
        :param option: option name string
        :param convert: boolean flag
        :return: value
        """
        value = self._cfg.get(section, option)
        return self._str2any(value) if convert else value

    @overload
    def write(self, section: str, option: str, value: Any) -> None:
        """
        :param section: section name string
        :param option: option name string
        :param value: value to be write in the section option
        :return: None
        """
        ...

    @overload
    def write(self, dictionary: dict) -> None:
        """
        :param dictionary: dictionary of values in format
            {section: {option: value}}
        :return: None
        """
        ...

    def write(self, *args) -> None:
        """
        write value or dict to the configfile.

        Also writes values to the configparser
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
        convert any value to the string.

        There is a specific rules for converting
        dict, tuple and list

        :param value: value for converting
        :return: value string
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
        get value from the configfile

        :param section: section name string
        :param option: option name string
        :return: value from the configfile
        """

        config = configparser.ConfigParser()
        config.read(self._filepath)
        return self._str2any(config.get(section, option))

    def _str2any(self, value: str) -> Any:
        """
        convert string value to the any.

        If there are no templates to convert, then returns
        the value 'as is' as a string.

        :param value: string value
        :return: converted value
        """

        def val2any(val: str) -> int | float | str | bool:
            if self.INT_PATTERN.match(val) is not None:
                return int(val)

            elif (self.FLOAT_PATTERN.match(val) or
                  self.EFLOAT_PATTERN.match(val)) is not None:
                return float(val)

            elif val in ('True', 'False'):
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
        :return: ConfigParser
        """
        return self._cfg

    @property
    def filepath(self):
        """
        :return: path to the configfile
        """
        return self._filepath

import io
import re
import configparser
from pathlib import Path
from typing import Any


class RWConfig(object):
    _list_delimiter = ','
    _list_pattern = re.compile(_list_delimiter)
    _tuple_delimiter = ';'
    _tuple_pattern = re.compile(_tuple_delimiter)

    _int_pattern = re.compile('^\d+$')
    _float_pattern = re.compile('^\d+\.\d+$')
    _efloat_pattern = re.compile('^\d\.\d+[eE][+-]\d+$')

    _filename_pattern = re.compile('\w+.ini$')

    def __init__(self, config_path: Path = None):

        self.config_path = config_path if config_path is not None else Path()
        self._config = configparser.ConfigParser()
        self.upd_config()

    def upd_config(self) -> None:
        self._config = configparser.ConfigParser()
        if self.config_path is not None:
            self.change_path(self.config_path)

    def change_path(self, config_path: Path) -> None:

        if self._filename_pattern.match(
                config_path.name) is None:
            raise ValueError(
                f'Указанный путь не ведет к *.ini файлу: {config_path}'
            )

        self.config_path = config_path
        if not self.config_path.parent.exists():
            self.config_path.parent.mkdir(parents=True)
        self._config = self.get_config()

    def get_config(
            self, from_: str = 'file') -> configparser.ConfigParser:

        if from_ == 'file':
            config = configparser.ConfigParser()
            if self.config_path.exists():
                config.read(self.config_path)
            else:
                with io.open(self.config_path, 'w') as cnfg_file:
                    config.write(cnfg_file)
            return config
        elif from_ == 'self':
            return self._config
        else:
            raise ValueError

    def set(self, section: str, option: str, value: Any) -> None:
        self._config.set(section, option, self._convert_any2str(value))

    def apply_changes(self) -> None:
        with io.open(self.config_path, 'w') as cnfg_file:
            self._config.write(cnfg_file)

    def get(self, section: str, option: str, convert: bool = True) -> Any:
        value = self._config.get(section, option)
        if convert:
            return self._convert2any(value)
        else:
            return value

    def write(self, section: str, option: str, value: Any) -> None:

        conv_value = self._convert_any2str(value)
        config = configparser.ConfigParser()
        config.read(self.config_path)
        config.set(section, option, conv_value)
        self.set(section, option, conv_value)
        with io.open(self.config_path, 'w') as cnfg_file:
            config.write(cnfg_file)

    def write_dict(self, dictionary: dict) -> None:

        conv_dict = {}
        for section, item in dictionary.items():
            conv_dict[section] = {}
            for option, raw_value in item.items():
                conv_dict[section][option] = self._convert_any2str(raw_value)
        self._config = configparser.ConfigParser()
        self._config.read_dict(conv_dict)
        with io.open(self.config_path, 'w') as cnfg_file:
            self._config.write(cnfg_file)

    def _convert_any2str(self, value) -> str:

        if isinstance(value, dict):
            conv_value = self._list_delimiter.join(
                [self._tuple_delimiter.join(
                    map(self._convert_val2str, item)
                ) for item in value.items()]
            )
        elif isinstance(value, tuple):
            conv_value = self._tuple_delimiter.join(
                map(self._convert_val2str, value)
            )
        elif isinstance(value, list):
            conv_value = self._list_delimiter.join(
                map(self._convert_val2str, value)
            )
        else:
            conv_value = self._convert_val2str(value)

        return conv_value

    @staticmethod
    def _convert_val2str(value: int | float | str) -> str:
        return str(value)

    def read(self, section: str, option: str) -> str:

        config = configparser.ConfigParser()
        config.read(self.config_path)
        return self._convert2any(config.get(section, option))

    def _convert2any(self, value: str) -> Any:

        if (self._tuple_pattern.search(value) is not None and
                self._list_pattern.search(value) is not None):
            raw_dict = [item.split(self._tuple_delimiter)
                        for item in value.split(self._list_delimiter)]
            return {
                self._convert_val2any(raw_key):
                    self._convert_val2any(raw_val)
                for (raw_key, raw_val) in raw_dict
            }

        elif self._tuple_pattern.search(value) is not None:
            return tuple(map(self._convert_val2any,
                             value.split(self._tuple_delimiter)))
        elif self._list_pattern.search(value) is not None:
            return list(map(self._convert_val2any,
                            value.split(self._list_delimiter)))
        else:
            return self._convert_val2any(value)

    def _convert_val2any(self, value: str) -> int | float | str | bool:

        if self._int_pattern.match(value) is not None:
            return int(value)
        elif (self._float_pattern.match(value) is not None or
              self._efloat_pattern.match(value) is not None):
            return float(value)
        elif value in ('True', 'False'):
            return eval(value)
        else:
            return value

    def items(self) -> configparser.ConfigParser.items:
        return self._config.items()

import io
import re
import configparser
from pathlib import Path
from typing import overload, Any

from ._rwf_exception import FilepathPatternError


__all__ = ['RWConfig']


class RWConfig(object):

    LIST_DELIMITER = ','
    LIST_PATTERN = re.compile(LIST_DELIMITER)
    TUPLE_DELIMITER = ';'
    TUPLE_PATTERN = re.compile(TUPLE_DELIMITER)

    INT_PATTERN = re.compile('^\d+$')
    FLOAT_PATTERN = re.compile('^\d+\.\d+$')
    EFLOAT_PATTERN = re.compile('^\d\.\d+[eE][+-]\d+$')

    FILENAME_PATTERN = re.compile('\w+.ini$')

    def __init__(self, filepath: Path | str):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        self._cfg_path = filepath
        self._cfg = configparser.ConfigParser()
        self.update_config()

    def update_config(self) -> None:
        self.change_path(self._cfg_path)

    def change_path(self, filepath: Path | str) -> None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if self.FILENAME_PATTERN.match(
                filepath.name) is None:
            raise FilepathPatternError(
                self.FILENAME_PATTERN, filepath)

        self._cfg_path = filepath
        if not self._cfg_path.parent.exists():
            self._cfg_path.parent.mkdir(parents=True)
        self._cfg = self.get_config_from_path()

    def get_config_from_path(self) -> configparser.ConfigParser:

        config = configparser.ConfigParser()
        if self._cfg_path.exists():
            config.read(self._cfg_path)
        else:
            with io.open(self._cfg_path, 'w') as cfg_file:
                config.write(cfg_file)
        return config

    def set(self, section: str, option: str, value: Any) -> None:
        self._cfg.set(section, option, self._convert_any2str(value))

    def apply_changes(self) -> None:
        with io.open(self._cfg_path, 'w') as cfg_file:
            self._cfg.write(cfg_file)

    def get(self, section: str, option: str, convert: bool = True) -> Any:
        value = self._cfg.get(section, option)
        return self._convert2any(value) if convert else value

    @overload
    def write(self, section: str, option: str, value: Any) -> None:
        ...

    @overload
    def write(self, dictionary: dict):
        ...

    def write(self, *args) -> None:

        match args:
            case (str() as section, str() as option, value):
                conv_value = self._convert_any2str(value)
                config = configparser.ConfigParser()
                config.read(self._cfg_path)
                config.set(section, option, conv_value)
                self.set(section, option, conv_value)
                with io.open(self._cfg_path, 'w') as cnfg_file:
                    config.write(cnfg_file)

            case (dict() as dictionary,):
                conv_dict = {}
                for section, item in dictionary.items():
                    conv_dict[section] = {}
                    for option, raw_value in item.items():
                        conv_dict[section][option] = self._convert_any2str(raw_value)
                self._cfg.read_dict(conv_dict)
                with io.open(self._cfg_path, 'w') as cnfg_file:
                    self._cfg.write(cnfg_file)

            case _:
                raise TypeError('Wrong args for write method')

    def _convert_any2str(self, value) -> str:

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

        config = configparser.ConfigParser()
        config.read(self._cfg_path)
        return self._convert2any(config.get(section, option))

    def _convert2any(self, value: str) -> Any:

        def str2any(val: str) -> int | float | str | bool:
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
            return {str2any(raw_key): str2any(raw_val)
                    for (raw_key, raw_val) in raw_dict}

        elif self.TUPLE_PATTERN.search(value) is not None:
            return tuple(str2any(v) for v in
                         value.split(self.TUPLE_DELIMITER))

        elif self.LIST_PATTERN.search(value) is not None:
            return [str2any(v) for v in
                    value.split(self.LIST_DELIMITER)]

        else:
            return str2any(value)

    @property
    def config(self):
        return self._cfg

    @property
    def path(self):
        return self._cfg_path

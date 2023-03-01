"""
====================================
RWConfig (:mod:`pyiak_instr.rwfile`)
====================================

The module provides class for work with config file
"""
import io
from pathlib import Path
from configparser import ConfigParser
from typing import overload, Any

from ._core import RWFile
from ..utilities import StringEncoder


__all__ = ["RWConfig"]


class RWConfig(RWFile[ConfigParser]):
    """
    Class for reading and writing to the configfile as *.ini.

    Include autoencoder for values.

    Parameters
    ----------
    filepath: Path | str
        path to config file *.ini.
    """

    ALLOWED_SUFFIXES = {".ini"}

    def __init__(self, filepath: Path | str):
        super().__init__(filepath, self._get_parser(filepath))

    def close(self) -> None:
        pass

    def commit(self) -> None:
        """
        Write configparser to the configfile.

        Used for save changes which created by .set method.
        """
        with io.open(self._fp, "w", encoding="cp1251") as file:
            self._api.write(file)

    def drop_changes(self) -> None:
        """
        Drop changes by reading config from `filepath`.
        """
        self.close()
        self._api = self._get_parser(self._fp)

    def get(self, section: str, option: str, convert: bool = True) -> Any:
        """
        Get value from the configparser.

        If convert is True, then the value will try to convert to some type
        by StringEncoder.
        If convert is False the value will be returned 'as is' as a string.

        Parameters
        ----------
        section: str
            section name.
        option: str
            option name.
        convert: bool, default=True
            convert the resulting value from str.

        Returns
        -------
        Any
            resulting value from configfile.
        """
        value = self._api.get(section, option)
        if convert:
            return StringEncoder.decode(value)
        return value

    @overload
    def set(
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
        convert: bool, default=True
            convert the `value` to str by StringEncoder.
        """

    @overload
    def set(
        self, section: str, options: dict[str, Any], convert: bool = True
    ) -> None:
        """
        Parameters
        ----------
        section: str
            section name.
        options: dict[str, Any]
            dictionary of values in format {option: value}.
        convert: bool, default=True
            convert the `value` to str by StringEncoder.
        """

    @overload
    def set(
        self, sections: dict[str, dict[str, Any]], convert: bool = True
    ) -> None:
        """
        Parameters
        ----------
        sections: dict[str, dict[str, Any]]
            dictionary of values in format {section: {option: value}}.
        convert: bool
            convert the `value` to str by StringEncoder.
        """

    def set(self, *args: Any, convert: bool = True, **kwargs: Any) -> None:
        """
        Write value or dict to the configfile.

        write(section: str, option: str, value: Any) -> None.
        write(section: str, options: dict[str, Any]) -> None.
        write(sections: dict[str, dict[str, Any]]) -> None.

        Parameters
        ----------
        *args: Any
            arguments for sets value to section, option.
        convert: bool, default=True
            convert the resulting value to str by StringEncoder.
        **kwargs: Any
            for mypy compatibility.
        """
        if len(kwargs):
            raise ValueError("kwargs cannot used here")

        def convert_value(value: Any) -> Any:
            if convert:
                value = StringEncoder.encode(value)
            return value

        match args:
            case (str() as sec, str() as opt, val):
                set_dict = {sec: {opt: convert_value(val)}}

            case (str() as sec, dict() as opts):
                set_dict = {
                    sec: {o: convert_value(v) for o, v in opts.items()}
                }

            case (dict() as secs,):
                set_dict = {
                    s: {o: convert_value(v) for o, v in opts.items()}
                    for s, opts in secs.items()
                }

            case _:
                raise TypeError(f"invalid arguments {args}")

        self._api.read_dict(set_dict)

    @staticmethod
    def _get_parser(filepath: Path | str) -> ConfigParser:
        """
        Read config from `filepath`.

        If the config on the specified path does not exist,
        creates an empty config file.

        Parameters
        ----------
        filepath: Path | str
            path to the parser.

        Returns
        -------
        configparser.ConfigParser
            config contains settings from file path.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        cfg = ConfigParser()
        if filepath.exists():
            cfg.read(filepath)
        else:
            with io.open(filepath, "w", encoding="cp1251") as file:
                cfg.write(file)
        return cfg

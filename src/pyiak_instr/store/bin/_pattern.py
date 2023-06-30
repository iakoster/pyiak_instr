"""Private module of ``pyiak_instr.store.bin.types``"""
from __future__ import annotations
from pathlib import Path
from typing import (
    Any,
    Self,
    TypeVar,
    cast,
)

from ...exceptions import NotConfiguredYet
from ...types import (
    Additions,
    SurPattern,
    Pattern,
    WritableMixin,
)
from ...rwfile import RWConfig
from ._container import Container
from ._struct import (
    Field,
    Struct,
)


__all__ = [
    "FieldPattern",
    "StructPattern",
    "ContainerPattern",
    "ContinuousStructPattern",
]


FieldT = TypeVar("FieldT", bound=Field)
StructT = TypeVar("StructT", bound=Struct[Any])
ContainerT = TypeVar("ContainerT", bound=Container[Any, Any, Any])


class FieldPattern(Pattern[FieldT]):
    """
    Represent abstract class of pattern for bytes struct (field).
    """

    @property
    def is_dynamic(self) -> bool:
        """
        Returns
        -------
        bool
            True if the pattern can be interpreted as a dynamic,
            otherwise - False.
        """
        return self.bytesize <= 0

    @property
    def bytesize(self) -> int:
        """
        Returns
        -------
        int
            size of the field in bytes.
        """
        if "bytes_expected" in self._kw:
            return cast(int, self._kw["bytes_expected"])

        start = cast(int, self._kw["start"]) if "start" in self._kw else 0
        stop = cast(int, self._kw["stop"]) if "stop" in self._kw else None

        if stop is None:
            if start < 0:
                return -start
        elif start >= 0 and stop > 0 or start < 0 and stop < 0:
            return stop - start
        return 0


FieldtPatternT = TypeVar("FieldtPatternT", bound=FieldPattern[Any])


class StructPattern(SurPattern[StructT, FieldtPatternT]):
    """
    Represent abstract class of pattern for bytes struct (storage).
    """

    def _modify_additions(
        self, additions: Additions | None = None
    ) -> Additions:
        """
        Modify additions for target and sub-patterns.

        Parameters
        ----------
        additions : Additions | None, default=None
            additions instance. If None - Additions from class will be used.

        Returns
        -------
        Additions
            additions instance.
        """
        add = super()._modify_additions(additions)
        for name in self._sub_p:
            add.lower(name).current["name"] = name
        return add

    def _get_parameters(self, additions: Additions) -> dict[str, Any]:
        """
        Get joined additions with pattern parameters.

        Also initialize fields for struct.

        Parameters
        ----------
        additions : dict[str, Any]
            additional initialization parameters.

        Returns
        -------
        dict[str, Any]
            joined parameters.
        """
        parameters = super()._get_parameters(additions=additions)
        parameters["fields"] = {
            n: p.get(additions=additions.lower(n))
            for n, p in self._sub_p.items()
        }
        return parameters


StructPatternT = TypeVar("StructPatternT", bound=StructPattern[Any, Any])


class ContinuousStructPattern(
    StructPattern[StructT, FieldtPatternT],
):
    """
    Represents methods for configure continuous storage.

    It's means `start` of the field is equal to `stop` of previous field
    (e.g. without gaps in content).
    """

    def _modify_additions(
        self, additions: Additions | None = None
    ) -> Additions:
        """
        Modify additions for target and sub-patterns.

        Parameters
        ----------
        additions : Additions | None, default=None
            additions instance. If None - Additions from class will be used.

        Returns
        -------
        Additions
            additions instance.
        """
        add = super()._modify_additions(additions)
        dyn_name = self._modify_before_dyn(add)
        if dyn_name is not None:
            self._modify_after_dyn(dyn_name, add)
        return add

    def _modify_before_dyn(self, additions: Additions) -> str | None:
        """
        Modify `additions` up to dynamic field.

        Parameters
        ----------
        additions : Additions
            additional kwargs for sub-pattern object.

        Returns
        -------
        str | None
            name of the dynamic field. If None - there is no dynamic field.
        """
        start = 0
        for name, pattern in self._sub_p.items():
            additions.lower(name).current.update(start=start)
            if pattern.is_dynamic:
                return name
            start += pattern.bytesize
        return None

    def _modify_after_dyn(self, dyn_name: str, additions: Additions) -> None:
        """
        Modify `sub_additions` from dynamic field to end.

        Parameters
        ----------
        dyn_name : str
            name of the dynamic field.
        additions : SubPatternAdditions
            additional kwargs for sub-pattern object.

        Raises
        ------
        TypeError
            if there is tow dynamic fields.
        AssertionError
            if for some reason the dynamic field is not found.
        """
        start = 0
        for name in list(self._sub_p)[::-1]:
            pattern = self._sub_p[name]

            if pattern.is_dynamic:
                if name == dyn_name:
                    additions.lower(name).current.update(
                        stop=start if start != 0 else None
                    )
                    return
                raise TypeError("two dynamic field not allowed")

            start -= pattern.bytesize
            additions.lower(name).current.update(start=start)

        raise AssertionError("dynamic field not found")


class ContainerPattern(SurPattern[ContainerT, StructPatternT], WritableMixin):
    """
    Represent pattern for bytes storage.
    """

    def configure(self, **patterns: StructPatternT) -> Self:
        """
        Configure bytes storage pattern.

        Only one pattern allowed.

        Parameters
        ----------
        **patterns : PatternT
            dictionary of patterns where key is a pattern name.

        Returns
        -------
        Self
            self instance.

        Raises
        ------
        TypeError
            if patterns more than one.

        See Also
        --------
        super().configure
        """
        if len(patterns) > 1:
            raise TypeError(
                f"only one storage pattern allowed, got {len(patterns)}"
            )
        return super().configure(**patterns)

    def write(self, path: Path) -> None:
        """
        Write pattern configuration to config file.

        Parameters
        ----------
        path : Path
            path to config file.

        Raises
        ------
        NotConfiguredYet
            is patterns is not configured yet.
        """
        if len(self._sub_p) == 0:
            raise NotConfiguredYet(self)

        pattern: StructPatternT
        ((name, pattern),) = self._sub_p.items()
        pars = {
            "_": self.__init_kwargs__(),
            name: pattern.__init_kwargs__(),
        }
        pars.update(
            {
                n: pattern.sub_pattern(n).__init_kwargs__()
                for n in pattern.sub_pattern_names
            }
        )

        with RWConfig(path) as cfg:
            if cfg.api.has_section(name):
                cfg.api.remove_section(name)
            cfg.set({name: pars})
            cfg.commit()

    @classmethod
    def read(cls, path: Path, name: str = "") -> Self:
        """
        Read init kwargs from `path` and initialize class instance.

        Parameters
        ----------
        path : Path
            path to the file.
        name : str, default=''
            name to search required pattern in file.

        Returns
        -------
        Self
            initialized self instance.

        Raises
        ------
        ValueError
            if `name` is empty string.
        """
        if len(name) == 0:
            raise ValueError("empty name not allowed")

        with RWConfig(path) as cfg:
            opts = cfg.api.options(name)
            opts.pop(opts.index("_"))
            opts.pop(opts.index(name))

            field_type = cls._sub_p_type.sub_pattern_type()
            struct_pattern = cls._sub_p_type(**cfg.get(name, name)).configure(
                **{f: field_type(**cfg.get(name, f)) for f in opts}
            )
            return cls(**cfg.get(name, "_")).configure(
                **{name: struct_pattern}
            )

    def _get_parameters(self, additions: Additions) -> dict[str, Any]:
        """
        Get joined additions with pattern parameters.

        Also initialize struct object and add pattern reference to parameters.

        Parameters
        ----------
        additions : dict[str, Any]
            additional initialization parameters.

        Returns
        -------
        dict[str, Any]
            joined parameters.
        """
        parameters = super()._get_parameters(additions)

        ((name, sub_pattern),) = self._sub_p.items()
        parameters["struct"] = sub_pattern.get(
            additions=additions.lower(name)
        )
        parameters["pattern"] = self.copy()
        return parameters

    def _modify_additions(
        self, additions: Additions | None = None
    ) -> Additions:
        """
        Modify additions for target and sub-patterns.

        Parameters
        ----------
        additions : Additions | None, default=None
            additions instance. If None - Additions from class will be used.

        Returns
        -------
        Additions
            additions instance.
        """
        add = super()._modify_additions(additions)
        for name in self._sub_p:
            add.lower(name).current["name"] = name
        return add

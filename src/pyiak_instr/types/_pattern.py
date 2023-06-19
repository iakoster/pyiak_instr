"""Private module of ``pyiak_instr.types`` with pattern types."""
from __future__ import annotations
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

from ..exceptions import NotConfiguredYet


__all__ = [
    "Additions",
    "EditableMixin",
    "SurPattern",
    "Pattern",
    "WritableMixin",
]


T = TypeVar("T")
OptionsT = TypeVar("OptionsT")  # Pattern target options


# todo: tests
class Additions:
    """
    Additional kwargs container for pattern target.

    Parameters
    ----------
    current: dict[str, Any] | None, default=None
        additional kwargs for pattern target.
    lower: dict[str, Additions] | None, default=None
        additional kwargs containers for sub-pattern target.
    """

    def __init__(
        self,
        current: dict[str, Any] | None = None,
        lower: dict[str, Additions] | None = None,
    ):
        self.current: dict[str, Any] = {} if current is None else current
        self.lowers: dict[str, Additions] = {} if lower is None else lower

    def get_joined(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Get joined `kwargs` with current additions.

        Takes a copy of `kwargs`. `kwargs` have a lower priority than
        additions parameters.

        Parameters
        ----------
        kwargs : dict[str, Any]
            original kwargs from pattern.

        Returns
        -------
        dict[str, Any]
            joined parameters.
        """
        parameters = kwargs.copy()
        parameters.update(self.current)
        return parameters

    def lower(self, name: str) -> Additions:
        """
        Get additional kwargs container for sub-pattern.

        If container not exists - create new.

        Parameters
        ----------
        name : str
            sub-pattern name.

        Returns
        -------
        Additions
            kwargs container.
        """
        if name not in self.lowers:
            self.lowers[name] = Additions()
        return self.lowers[name]

    @property
    def lower_names(self) -> list[str]:
        """
        Returns
        -------
        list[str]
            names of lowers.
        """
        return list(self.lowers)


class Pattern(ABC, Generic[OptionsT]):
    """
    Represents protocol for patterns.

    Parameters
    ----------
    typename: str
        name of pattern target type.
    **parameters: Any
        parameters for target initialization.
    """

    _options: dict[str, type[OptionsT]]
    """pattern target options"""

    # todo: parameters to TypedDict (3.12: default dict[str, Any], another -
    #  specified)
    def __init__(self, typename: str, **parameters: Any) -> None:
        if typename not in self._options:
            raise KeyError(f"'{typename}' not in {set(self._options)}")

        self._tn = typename
        self._kw = parameters

    def get(
        self,  # pylint: disable=unused-argument
        additions: Additions = Additions(),
        **kwargs: Any,
    ) -> OptionsT:
        """
        Get initialized class instance with parameters from pattern and from
        `additions`.

        Parameters
        ----------
        additions: Additions, default=Additions()
            container with additional initialization parameters.
        **kwargs: Any
            ignored. Needed for backward compatibility.

        Returns
        -------
        OptionsT
            initialized target class.
        """
        return self._target(**self._get_parameters(additions))

    def _get_parameters(self, additions: Additions) -> dict[str, Any]:
        """
        Get joined additions with pattern parameters.

        Parameters
        ----------
        additions : dict[str, Any]
            additional initialization parameters.

        Returns
        -------
        dict[str, Any]
            joined parameters.
        """
        return additions.get_joined(self._kw)

    @property
    def typename(self) -> str:
        """
        Returns
        -------
        str
            name of pattern target type.
        """
        return self._tn

    @property
    def _target(self) -> type[OptionsT]:
        """
        Get target class by `typename` or default if `typename` not in
        options.

        Returns
        -------
        type[OptionsT]
            target class.
        """
        return self._options[self._tn]

    def __init_kwargs__(self) -> dict[str, Any]:
        return dict(typename=self._tn, **self._kw)

    def __getitem__(self, name: str) -> Any:
        """Get parameter value"""
        return self._kw[name]

    def __new__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if not hasattr(cls, "_options"):
            raise AttributeError(
                f"'{cls.__name__}' object has no attribute '_options'"
            )
        return object.__new__(cls)


PatternT = TypeVar("PatternT", bound=Pattern[Any])


# todo: right mixin (self: Pattern)
class EditableMixin:
    """
    Represents abstract class with methods to edit parameters.
    """

    _kw: dict[str, Any]

    def pop(self, key: str) -> Any:
        """
        Extract parameter with removal.

        Parameters
        ----------
        key: str
            parameter name.

        Returns
        -------
        Any
            parameter value.
        """
        return self._kw.pop(key)

    def __setitem__(self, name: str, value: Any) -> None:
        """Change/add pattern to pattern."""
        self._kw[name] = value


# todo: right mixin (self: Pattern)
class WritableMixin(ABC):
    """
    Represents protocol for patterns which supports read/write.
    """

    @abstractmethod
    def write(self, path: Path) -> None:
        """
        Write pattern init kwargs to the file at the `path`.

        Parameters
        ----------
        path: Path
            path to file.
        """

    @classmethod
    @abstractmethod
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
        """


class SurPattern(
    Pattern[OptionsT],
    Generic[OptionsT, PatternT],
):
    """
    Represent pattern which consist of other patterns.

    Used to create an instance of a compound object (i.e. the object contains
    other objects generated by the sub-patterns)

    Parameters
    ----------
    typename: str
        name of pattern target type.
    **parameters: Any
        parameters for target initialization.
    """

    _sub_p_type: type[PatternT]
    """sub pattern type for initializing in class."""

    def __init__(self, typename: str, **parameters: Any) -> None:
        super().__init__(typename, **parameters)
        self._sub_p: dict[str, PatternT] = {}

    def configure(self, **patterns: PatternT) -> Self:
        """
        Configure storage pattern with other patterns.

        Parameters
        ----------
        **patterns : PatternT
            dictionary of patterns where key is a pattern name.

        Returns
        -------
        Self
            self instance.
        """
        # todo: check auto pars for subs
        self._sub_p = patterns
        return self

    def get(
        self,
        additions: Additions = Additions(),
        **kwargs: Any,
    ) -> OptionsT:
        """
        Get initialized class instance with parameters from pattern and from
        `additions`.

        Parameters
        ----------
        additions: Additions, default=Additions()
            container with additional initialization parameters.
        **kwargs: Any
            ignored. Needed for backward compatibility.

        Returns
        -------
        OptionsT
            initialized target class.

        Raises
        ------
        NotConfiguredYet
            if sub-patterns list is empty.
        """
        if len(self._sub_p) == 0:
            raise NotConfiguredYet(self)
        self._modify_additions(additions)
        return super().get(additions=additions, **kwargs)

    def sub_pattern(self, name: str) -> PatternT:
        """
        Get sub-pattern by `name`.

        Parameters
        ----------
        name : str
            sub-pattern name.

        Returns
        -------
        PatternT
            sub-pattern.
        """
        return self._sub_p[name]

    @classmethod
    def sub_pattern_type(cls) -> type[PatternT]:
        """
        Returns
        -------
        type[PatternT]
            sub-pattern type.
        """
        return cls._sub_p_type

    def _modify_additions(self, additions: Additions) -> None:
        """
        Modify additions for target and sub-patterns.

        Parameters
        ----------
        additions : Additions
            additions instance.
        """

    @property
    def sub_pattern_names(self) -> list[str]:
        """
        Returns
        -------
        list[str]
            list of sub-pattern names.
        """
        return list(self._sub_p)

    def __new__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if not hasattr(cls, "_sub_p_type"):
            raise AttributeError(
                f"'{cls.__name__}' object has no attribute '_sub_p_type'"
            )
        return super().__new__(  # type: ignore[no-any-return, misc]
            cls, *args, **kwargs
        )

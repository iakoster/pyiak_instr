"""Private module of ``pyiak_instr.types`` with pattern types."""
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

from ..exceptions import NotConfiguredYet
from ..utilities import split_complex_dict


__all__ = [
    "EditablePatternABC",
    "MetaPatternABC",
    "PatternABC",
    "WritablePatternABC",
]


OptionsT = TypeVar("OptionsT")  # Pattern target options
PatternT = TypeVar("PatternT", bound="PatternABC[Any]")


class PatternABC(ABC, Generic[OptionsT]):
    """
    Represents protocol for patterns.

    Parameters
    ----------
    typename: str
        name of pattern target type.
    **kwargs: Any
        parameters for target initialization.
    """

    _options: dict[str, type[OptionsT]]
    """pattern target options"""

    _required_init_parameters: set[str] = set()

    # todo: parameters to TypedDict
    def __init__(self, typename: str, **parameters: Any) -> None:
        if typename not in self._options:
            raise KeyError(f"'{typename}' not in {set(self._options)}")
        if not self._required_init_parameters.issubset(parameters):
            raise TypeError(
                f"{self._required_init_parameters.difference(parameters)} "
                "not represented in parameters"
            )

        self._tn = typename
        self._kw = parameters

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> OptionsT:
        """
        Get initialized class instance with parameters from pattern and from
        `additions`.

        Parameters
        ----------
        changes_allowed: bool, default = False
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection, otherwise the
            `additions` take precedence.
        **additions: Any
            additional initialization parameters.

        Returns
        -------
        OptionsT
            initialized target class.
        """
        return self._target(
            **self._get_parameters_dict(changes_allowed, additions)
        )

    def _get_parameters_dict(
        self,
        changes_allowed: bool,
        additions: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get joined additions with pattern parameters.

        Parameters
        ----------
        changes_allowed : bool
            allows situations where keys from the pattern overlap with kwargs.
        additions : dict[str, Any]
            additional initialization parameters.

        Returns
        -------
        dict[str, Any]
            joined parameters.

        Raises
        ------
        SyntaxError
            if changes not allowed but parameter repeated in `additions`.
        TypeError
            if parameter in `additions` and pattern, but it must be to set
            automatically.
        """
        parameters = self._kw.copy()
        if len(additions) != 0:
            if not changes_allowed:
                intersection = set(parameters).intersection(set(additions))
                if len(intersection):
                    raise SyntaxError(
                        "keyword argument(s) repeated: "
                        f"{', '.join(intersection)}"
                    )

            parameters.update(additions)
        return parameters

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
        Get target class by `type_name` or default if `type_name` not in
        options.

        Returns
        -------
        type[OptionsT]
            target class.
        """
        return self._options[self._tn]

    def __init_kwargs__(self) -> dict[str, Any]:
        return dict(typename=self._tn, **self._kw)

    def __contains__(self, name: str) -> bool:
        """Check that parameter in Pattern by name."""
        return name in self._kw

    def __eq__(self, other: object) -> bool:
        """
        Compare `__init_kwargs__` in two objects.

        Beware: not supports numpy arrays in __init_kwargs__.
        """
        if hasattr(other, "__init_kwargs__"):
            return (  # type: ignore[no-any-return]
                self.__init_kwargs__() == other.__init_kwargs__()
            )
        return False

    def __getitem__(self, name: str) -> Any:
        """Get parameter value"""
        return self._kw[name]


class EditablePatternABC(ABC):
    """
    Represents abstract class with methods to edit parameters.
    """

    _kw: dict[str, Any]

    def add(self, key: str, value: Any) -> None:
        """
        Add new parameter to the pattern.

        Parameters
        ----------
        key : str
            new parameter name.
        value : Any
            new parameter value.

        Raises
        ------
        KeyError
            if parameter name is already exists.
        """
        if key in self._kw:
            raise KeyError(f"parameter '{key}' in pattern already")
        self._kw[key] = value

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
        """Change value of the existed parameter."""
        if name not in self._kw:
            raise KeyError(f"'{name}' not in parameters")
        self._kw[name] = value


class WritablePatternABC(ABC):
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
    def read(cls, path: Path, *keys: str) -> Self:
        """
        Read init kwargs from `path` and initialize class instance.

        Parameters
        ----------
        path : Path
            path to the file.
        *keys : str
            keys to search required pattern in file.

        Returns
        -------
        Self
            initialized self instance.
        """


class MetaPatternABC(  # todo: rename to antonym of sub
    PatternABC[OptionsT],
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
    name: str
        name of pattern meta-object format.
    **kwargs: Any
        parameters for target initialization.
    """

    _sub_p_type: type[PatternT]
    """sub pattern type for initializing in class."""

    _sub_p_par_name: str
    """is the name of the parameter that names the sub-pattern in the
    meta-object"""

    def __init__(self, typename: str, name: str, **kwargs: Any):
        if not hasattr(self, "_sub_p_par_name"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute "
                "'_sub_p_par_name'"
            )
        super().__init__(typename, name=name, **kwargs)
        self._name = name
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
        self, changes_allowed: bool = False, **additions: Any
    ) -> OptionsT:
        """
        Get initialized class instance with parameters from pattern and from
        `additions`.

        Parameters
        ----------
        changes_allowed: bool, default = False
            allows situations where keys from the pattern overlap with kwargs.
            If False, it causes an error on intersection, otherwise the
            `additions` take precedence.
        **additions: Any
            additional initialization parameters. Those keys that are
            separated by "__" will be defined as parameters for other
            patterns target, otherwise for the storage target.

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
        for_subs, for_meta = split_complex_dict(
            additions, without_sep="other"
        )
        # todo: check that `for_subs` intersection all `sub_p`
        return super().get(
            changes_allowed,
            **for_meta,
            **self._get_subs(changes_allowed, for_subs),
        )

    def _get_subs(
        self, ch_a: bool, for_subs: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:  # ch_a - is changes allowed
        """
        Get dictionary of sub-pattern objects.

        Parameters
        ----------
        ch_a : bool
            if True allows situations where keys from the pattern overlap
            with kwargs.
        for_subs : dict[str, dict[str, Any]]
            additional kwargs for sub-pattern objects if format
            {FIELD: {PARAMETER: VALUE}}.

        Returns
        -------
        dict[str, Any]
            dictionary of sub-pattern objects.
        """
        # todo: fix crutch? (changes_allowed always True for sub-patterns).
        #  if needed for editing additions for sub-patterns in meta-pattern.
        for_subs = self._modify_all(ch_a, for_subs)
        return {
            self._sub_p_par_name: {
                n: s.get(True, **self._modify_each(ch_a, n, for_subs[n]))
                for n, s in self._sub_p.items()
            }
        }

    # pylint: disable=unused-argument
    def _modify_all(
        self, changes_allowed: bool, for_subs: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Modify additional kwargs for sub-pattern objects.

        Parameters
        ----------
        changes_allowed : bool
            if True allows situations where keys from the pattern overlap
            with kwargs.
        for_subs : dict[str, dict[str, Any]]
            additional kwargs for sub-pattern object if format
            {FIELD: {PARAMETER: VALUE}}.

        Returns
        -------
        dict[str, dict[str, Any]]
            modified additional kwargs for sub-pattern object.
        """
        return {
            k: (for_subs[k] if k in for_subs else {}) for k in self._sub_p
        }

    # pylint: disable=unused-argument
    def _modify_each(
        self,
        changes_allowed: bool,
        name: str,
        for_sub: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Modify additional kwargs for one sub-pattern object.

        Parameters
        ----------
        changes_allowed : bool
            if True allows situations where keys from the pattern overlap
            with kwargs.
        name: str
            name of field
        for_sub : dict[str, Any]
            additional kwargs for sub-pattern object if format
            {PARAMETER: VALUE}.

        Returns
        -------
        dict[str, Any]
            modified additional kwargs for one sub-pattern object.
        """
        return for_sub

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            name of pattern storage.
        """
        return self._name

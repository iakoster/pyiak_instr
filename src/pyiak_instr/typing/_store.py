"""Private module of ``pyiak_instr`` with types of store module."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Self, TypeVar


__all__ = [
    "BytesFieldABC",
    "PatternABC",
    "PatternStorageABC",
    "WritablePatternABC",
]


ParametersT = TypeVar("ParametersT")
StorageT = TypeVar("StorageT")
PatternT = TypeVar("PatternT")
PatternTargetOptionsT = TypeVar("PatternTargetOptionsT")


# pylint: disable=too-few-public-methods
class BytesFieldABC(ABC, Generic[StorageT, ParametersT]):
    """
    Abstract base class parser for byte fields.

    Parameters
    ----------
    storage : StorageT
        storage of fields.
    name : str
        field name.
    parameters : ParametersT
        parameters instance.
    """

    def __init__(
        self, storage: StorageT, name: str, parameters: ParametersT
    ) -> None:
        self._s = storage
        self._name = name
        self._p = parameters

    @property
    def parameters(self) -> ParametersT:
        """
        Returns
        -------
        FieldT
            parameters instance.
        """
        return self._p


class PatternABC(ABC, Generic[PatternTargetOptionsT]):
    """
    Represents protocol for patterns.

    Parameters
    ----------
    typename: str
        name of pattern target type.
    **kwargs: Any
        parameters for target initialization.
    """

    _target_options: dict[str, type[PatternTargetOptionsT]]
    _target_default: type[PatternTargetOptionsT]

    def __init__(self, typename: str, **parameters: Any) -> None:
        self._tn = typename
        self._kw = parameters

    @abstractmethod
    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> PatternTargetOptionsT:
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
        PatternTargetOptionsT
            initialized target class.
        """

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
    def _target(self) -> type[PatternTargetOptionsT]:
        """
        Get target class by `type_name` or default if `type_name` not in
        options.

        Returns
        -------
        type[PatternTargetOptionsT]
            target class.
        """
        return self._target_options.get(self._tn, self._target_default)

    def __init_kwargs__(self) -> dict[str, Any]:
        return dict(typename=self._tn, **self._kw)


# todo: EditablePattern


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


class PatternStorageABC(
    PatternABC[PatternTargetOptionsT],
    Generic[PatternTargetOptionsT, PatternT],
):
    """
    Represent pattern which consist of other patterns.

    Parameters
    ----------
    typename: str
        name of pattern target type.
    name: str
        ame of pattern storage format.
    **kwargs: Any
        parameters for target initialization.
    """

    _p: dict[str, PatternT]

    def __init__(self, typename: str, name: str, **kwargs: Any):
        super().__init__(typename, name=name, **kwargs)
        self._name = name
        self._p = {}

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
        self._p = patterns
        return self

    @abstractmethod
    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> PatternTargetOptionsT:
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
        PatternTargetOptionsT
            initialized target class.
        """

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            name of pattern storage.
        """
        return self._name

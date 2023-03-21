"""Private module of ``pyiak_instr.store`` for work with bytes."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from dataclasses import field as _field
from typing import Any, ClassVar, Self, Generator, Iterable

import numpy as np
import numpy.typing as npt

from ..core import Code
from ..rwfile import RWConfig
from ..exceptions import NotConfiguredYet
from ..utilities import BytesEncoder, split_complex_dict
from ..typing import (
    BytesFieldABC,
    PatternABC,
    PatternStorageABC,
    WritablePatternABC,
)


__all__ = [
    "BytesFieldStruct",
    "BytesField",
    "ContinuousBytesStorage",
    "BytesFieldPattern",
    "BytesStoragePattern",
]


@dataclass(frozen=True, kw_only=True)
class BytesFieldStruct:
    """
    Represents field parameters with values encoded in bytes.
    """

    start: int
    """the number of bytes in the message from which the fields begin."""

    fmt: Code
    """format for packing or unpacking the content.
    The word length is calculated from the format."""

    expected: int
    """expected number of words in the field. If less than 1, from
    the start byte to the end of the message."""

    order: Code = Code.BIG_ENDIAN
    """bytes order for packing and unpacking."""

    default: bytes = b""  # todo: to ContentType
    """default value of the field."""

    _stop: int | None = _field(default=None, repr=False)
    """stop byte index of field content."""

    ORDERS: ClassVar[dict[Code, str]] = BytesEncoder.ORDERS
    ALLOWED_CODES: ClassVar[set[Code]] = BytesEncoder.ALLOWED_CODES

    def __post_init__(self) -> None:
        BytesEncoder.check_fmt_order(self.fmt, self.order)

        if self.expected < 0:
            object.__setattr__(self, "expected", 0)
        stop = self.start + self.bytes_expected
        if not self.infinite and stop != 0:
            object.__setattr__(self, "_stop", stop)

    def decode(self, content: bytes) -> npt.NDArray[np.int_ | np.float_]:
        """
        Decode content from parent.

        Parameters
        ----------
        content: bytes
            content for decoding.

        Returns
        -------
        npt.NDArray[Any]
            decoded content.
        """
        return BytesEncoder.decode(content, fmt=self.fmt, order=self.order)

    def encode(self, content: Iterable[int | float] | int | float) -> bytes:
        """
        Encode array to bytes.

        Parameters
        ----------
        content : npt.ArrayLike
            content to encoding.

        Returns
        -------
        bytes
            encoded content.
        """
        return BytesEncoder.encode(content, fmt=self.fmt, order=self.order)

    def validate(self, content: bytes) -> bool:
        """
        Check the content for compliance with the field parameters.

        Parameters
        ----------
        content: bytes
            content for validating.

        Returns
        -------
        bool
            True - content is correct, False - not.
        """
        if self.infinite:
            if len(content) % self.word_size:
                return False
        elif len(content) != self.bytes_expected:
            return False
        return True

    @property
    def bytes_expected(self) -> int:
        """
        Returns
        -------
        int
            expected bytes for field. Returns 0 if field is infinite.
        """
        return self.expected * self.word_size

    @property
    def infinite(self) -> bool:
        """
        Returns
        -------
        bool
            Indicate that it is finite field.
        """
        return not self.expected

    @property
    def slice(self) -> slice:
        """
        Returns
        -------
        slice
            The range of bytes from the message belonging to the field.
        """
        return slice(self.start, self.stop)

    @property
    def stop(self) -> int | None:
        """
        Returns
        -------
        int | None
            The number of byte in the message to which the field stops.
        """
        return self._stop

    @property
    def word_size(self) -> int:
        """
        Returns
        -------
        int
            The length of the one word in bytes.
        """
        return BytesEncoder.get_bytesize(self.fmt)


class ContinuousBytesStorage:
    """
    Represents continuous storage where data storage in bytes.

    Continuous means that the storage fields must go continuously, that is,
    where one field ends, another must begin.

    Parameters
    ----------
    name: str
        name of storage.
    **fields: BytesFieldStruct
        fields of the storage. The kwarg Key is used as the field name.
    """

    def __init__(self, name: str, **fields: BytesFieldStruct):
        for f_name, field in fields.items():
            if not isinstance(field, BytesFieldStruct):
                raise TypeError(
                    "invalid type of %r: %s" % (f_name, type(field))
                )

        self._name = name
        self._f = fields
        self._c = bytes()

    def decode(self) -> dict[str, npt.NDArray[Any]]:
        """
        Iterate by fields end decode each.

        Returns
        -------
        dict[str, npt.NDArray[Any]]
            dictionary with decoded content where key is a field name.
        """
        decoded = {}
        for field in self:
            decoded[field.name] = field.decode()
        return decoded

    def extract(self, content: bytes) -> Self:
        """
        Extract fields from existing bytes content.

        Parameters
        ----------
        content: bytes
            new content.

        Returns
        -------
        Self
            self instance.
        """
        fields = {}
        for parser in self:
            new = content[parser.parameters.slice]
            if len(new):
                fields[parser.name] = new
        self.set(**fields)
        return self

    def set(self, **fields: Iterable[int | float] | int | float) -> Self:
        """
        Set content to the fields.

        Parameters
        ----------
        **fields: ArrayLike
            fields content

        Returns
        -------
        Self
            self instance

        Raises
        ------
        AttributeError
            if values of non-existent fields were passed or values of some
            fields were not passed.
        """

        diff = set()
        raw_diff = set(self._f).symmetric_difference(set(fields))
        for field in raw_diff:
            if field in self and (
                self[field].parameters.infinite or len(self[field]) != 0
            ):
                continue
            diff.add(field)

        if len(diff) != 0:
            raise AttributeError(
                "missing or extra fields were found: %r" % sorted(diff)
            )

        self._set(fields)
        return self

    def _set(
        self, fields: dict[str, Iterable[int | float] | int | float]
    ) -> None:
        """
        Set new content to the fields.

        Parameters
        ----------
        fields: fields: dict[str, ArrayLike]
            dictionary of new field content.
        """
        # todo: performance, replace content if content > 0 or
        #  if empty create new
        new_content = b""
        for field in self:
            if field.name in fields:
                new_content += self._get_new_field_content(
                    field, fields[field.name]
                )
            else:
                new_content += field.content
        self._c = new_content

    @staticmethod
    def _get_new_field_content(
        parser: BytesField,
        content: Iterable[int | float] | int | float,
    ) -> bytes:
        """
        Set new content to the field.

        Parameters
        ----------
        parser: BytesField
            field parser.
        content: ArrayLike
            new content.

        Returns
        -------
        bytes
            new field content

        Raises
        ------
        ValueError
            if new content is not correct for field.
        """
        if isinstance(content, bytes):
            new_content = content
        else:
            new_content = parser.parameters.encode(content)

        if not parser.parameters.validate(new_content):
            raise ValueError(
                "%r is not correct for %r"
                % (new_content.hex(" "), parser.name)
            )

        return new_content

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            content of the storage.
        """
        return self._c

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            name of the storage.
        """
        return self._name

    def __contains__(self, name: str) -> bool:
        """Check that field name exists."""
        return name in self._f

    def __getitem__(self, field: str) -> BytesField:
        """Get field parser."""
        return BytesField(self, field, self._f[field])

    def __iter__(self) -> Generator[BytesField, None, None]:
        """Iterate by field parsers."""
        for field in self._f:
            yield self[field]


# todo: up to this level all functions and properties from BytesField
# todo: __str__ method
# todo: typing - set content with bytes
class BytesField(BytesFieldABC[ContinuousBytesStorage, BytesFieldStruct]):
    """
    Represents parser for work with field content.
    """

    def decode(self) -> npt.NDArray[np.int_ | np.float_]:
        """
        Decode field content.

        Returns
        -------
        NDArray
            decoded content.
        """
        return self._p.decode(self.content)

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self._s.content[self._p.slice]

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            field name
        """
        return self._name

    @property
    def words_count(self) -> int:
        """
        Returns
        -------
        int
            Count of words in the field.
        """
        return len(self) // self._p.word_size

    def __bytes__(self) -> bytes:  # todo: tests
        """
        Returns
        -------
        bytes
            field content.
        """
        return self.content

    def __getitem__(
        self, index: int | slice
    ) -> np.int_ | np.float_:  # todo: tests
        """
        Parameters
        ----------
        index : int | slice
            word index.

        Returns
        -------
        int | float
            word value.
        """
        return self.decode()[index]  # type: ignore[return-value]

    def __iter__(self) -> Generator[int | float, None, None]:  # todo: tests
        """
        Yields
        ------
        int | float
            word value.
        """
        for item in self.decode():
            yield item

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            bytes count of the content
        """
        return len(self.content)


# todo: typehint - Generic. Create via generic for children.
# todo: to metaclass (because uses for generate new class)
# todo: wait to 3.12 for TypeVar with default type.
class BytesFieldPattern(PatternABC[BytesFieldStruct]):
    """
    Represents class which storage common parameters for field.
    """

    _target_options = {}
    _target_default = BytesFieldStruct

    def add(self, key: str, value: Any) -> None:
        """
        Add new parameter to the pattern.

        Parameters
        ----------
        key: str
            new parameter name.
        value: Any
            new parameter value.

        Raises
        ------
        KeyError
            if parameter name is already exists.
        """
        if key in self:
            raise KeyError("parameter in pattern already")
        self._kw[key] = value

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> BytesFieldStruct:
        """
        Get field initialized with pattern and `additions`.


        Parameters
        ----------
        changes_allowed : bool, default=False
            if False intersection between `additions` and pattern is
            prohibited.
        additions : Any
            additional arguments for field.

        Returns
        -------
        BytesFieldStruct
            initialized field.

        Raises
        ------
        SyntaxError
            if `changes_allowed` is False and pattern and `additions` has
            intersection.
        """
        if not changes_allowed:
            intersection = set(self._kw).intersection(set(additions))
            if len(intersection) > 0:
                raise SyntaxError(
                    f"keyword argument(s) repeated: {', '.join(intersection)}"
                )

        kwargs = self._kw.copy()
        kwargs.update(additions)
        return self._target(**kwargs)

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

    def __contains__(self, item: str) -> bool:
        return item in self._kw

    def __getitem__(self, parameter: str) -> Any:
        return self._kw[parameter]

    def __setitem__(self, parameter: str, value: Any) -> None:
        if parameter not in self:
            raise KeyError("%r not in parameters" % parameter)
        self._kw[parameter] = value


# todo: to metaclass? (because uses for generate new class)
class BytesStoragePattern(
    PatternStorageABC[ContinuousBytesStorage, BytesFieldPattern],
    WritablePatternABC,
):
    """
    Represents pattern for bytes storage.

    Parameters
    ----------
    typename: {'continuous'}
        typename of storage.
    **kwargs: Any
        parameters for storage initialization.
    """

    _target_options = {}
    _target_default = ContinuousBytesStorage

    def __init__(self, typename: str, name: str, **kwargs: Any):
        if typename != "continuous":
            raise ValueError(f"invalid typename: '{typename}'")
        super().__init__(typename, name, **kwargs)

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> ContinuousBytesStorage:
        """
        Get initialized storage.

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
        ContinuousBytesStorage
            initialized storage.

        Raises
        ------
        AssertionError
            if in some reason typename is invalid.
        NotConfiguredYet
            if patterns list is empty.
        """
        if len(self._p) == 0:
            raise NotConfiguredYet(self)

        for_field, for_storage = split_complex_dict(
            additions, without_sep="other"
        )

        if self._tn == "continuous":
            return self._get_continuous(for_storage, for_field)
        raise AssertionError(f"invalid typename: '{self._tn}'")

    def write(self, path: Path) -> None:
        """
        Write pattern configuration to config file.

        Parameters
        ----------
        path : Path
            path to config file.
        """
        pars = {self._name: self.__init_kwargs__()}
        for name, pattern in self._p.items():
            pars[name] = pattern.__init_kwargs__()

        with RWConfig(path) as cfg:
            if cfg.api.has_section(self._name):
                cfg.api.remove_section(self._name)
            cfg.set({self._name: pars})
            cfg.commit()

    def _get_continuous(
        self,
        for_storage: dict[str, Any],
        for_fields: dict[str, dict[str, Any]],
    ) -> ContinuousBytesStorage:
        """
        Get initialized continuous storage.

        Parameters
        ----------
        for_storage: dict[str, Any]:
            dictionary with parameters for storage in format
            {PARAMETER: VALUE}.
        for_fields: dict[str, dict[str, Any]]
            dictionary with parameters for fields in format
            {FIELD: {PARAMETER: VALUE}}.

        Returns
        -------
        ContinuousBytesStorage
            initialized storage.
        """
        storage_kw = self._kw
        if len(for_storage) != 0:
            storage_kw.update(for_storage)

        fields, inf = self._get_fields_before_inf(for_fields)
        if len(fields) != len(self._p):
            after, next_ = self._get_fields_after_inf(for_fields, inf)
            fields.update(after)
            object.__setattr__(fields[inf], "_stop", fields[next_].start)

        return self._target(**storage_kw, **fields)  # type: ignore[arg-type]

    def _get_fields_after_inf(
        self,
        fields_kw: dict[str, dict[str, Any]],
        inf: str,
    ) -> tuple[dict[str, BytesFieldStruct], str]:
        """
        Get the dictionary of fields that go from infinite field
        (not included) to end.

        Parameters
        ----------
        fields_kw : dict[str, dict[str, Any]]
            dictionary of kwargs for fields.
        inf : str
            name of infinite fields.

        Returns
        -------
        tuple[dict[str, BytesFieldStruct], str]
            fields - dictionary of fields from infinite (not included);
            next - name of next field after infinite.
        """
        rev_names = list(self._p)[::-1]
        fields, start, next_ = [], 0, ""
        for name in rev_names:
            if name == inf:
                break

            pattern = self._p[name]
            start -= pattern["expected"] * BytesEncoder.get_bytesize(
                pattern["fmt"]
            )
            field_kw = fields_kw[name] if name in fields_kw else {}
            field_kw.update(start=start)
            fields.append(
                (name, pattern.get(changes_allowed=True, **field_kw))
            )
            next_ = name

        return dict(fields[::-1]), next_

    def _get_fields_before_inf(
        self, fields_kw: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, BytesFieldStruct], str]:
        """
        Get the dictionary of fields that go up to and including the infinite
        field.

        Parameters
        ----------
        fields_kw : dict[str, dict[str, Any]]
            dictionary of kwargs for fields.

        Returns
        -------
        tuple[dict[str, BytesFieldStruct], str]
            fields - dictionary of fields up to infinite (include);
            inf - name of infinite field. Empty if there is no found.
        """
        fields, start, inf = {}, 0, ""
        for name, pattern in self._p.items():
            field_kw = fields_kw[name] if name in fields_kw else {}
            field_kw.update(start=start)

            # not supported, but it is correct behaviour
            # if len(set(for_field).intersection(pattern)) == 0:
            #     fields[name] = pattern.get(**for_field)
            fields[name] = pattern.get(changes_allowed=True, **field_kw)

            stop = fields[name].stop
            if stop is None:
                inf = name
                break
            start = stop

        return fields, inf

    @classmethod
    def read(cls, path: Path, *keys: str) -> Self:
        """
        Read init kwargs from `path` and initialize class instance.

        Parameters
        ----------
        path : Path
            path to the file.
        *keys : str
            keys to search required pattern in file. Must include only one
            argument - `name`.

        Returns
        -------
        Self
            initialized self instance.

        Raises
        ------
        TypeError
            if given invalid count of keys.
        """
        if len(keys) != 1:
            raise TypeError(f"given {len(keys)}, expect 1")
        (name,) = keys

        with RWConfig(path) as cfg:
            opts = cfg.api.options(name)
            opts.pop(opts.index(name))
            return cls(**cfg.get(name, name)).configure(
                **{f: BytesFieldPattern(**cfg.get(name, f)) for f in opts}
            )

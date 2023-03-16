"""Private module of ``pyiak_instr.store`` for work with bytes."""
from __future__ import annotations

import struct
from pathlib import Path
from dataclasses import dataclass
from dataclasses import field as _field
from typing import Any, ClassVar, Self, Generator

import numpy as np
import numpy.typing as npt

from ..core import Code
from ..rwfile import RWConfig
from ..exceptions import CodeNotAllowed, NotConfiguredYet
from ..utilities import BytesEncoder, split_complex_dict


__all__ = [
    "BytesField",
    "BytesFieldParser",
    "ContinuousBytesStorage",
    "BytesFieldPattern",
    "BytesStoragePattern",
]


@dataclass(frozen=True, kw_only=True)
class BytesField:
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
    VALUES: ClassVar[dict[Code, str]] = BytesEncoder.ALLOWED_CODES

    def __post_init__(self) -> None:
        if self.fmt not in BytesEncoder.ALLOWED_CODES:
            raise CodeNotAllowed(self.fmt)
        if self.order not in BytesEncoder.ORDERS:
            raise CodeNotAllowed(self.order)

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

    def encode(self, content: npt.ArrayLike) -> bytes:
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
        return struct.calcsize(self.VALUES[self.fmt])


# todo: up to this level all functions and properties from BytesField
# todo: __str__ method
# todo: parser via Generic[ContinuousBytesStorage, BytesField]
class BytesFieldParser:
    """
    Represents parser for work with field content.

    Parameters
    ----------
    storage: ContinuousBytesStorage
        storage of fields.
    name: str
        field name.
    field: BytesField
        field instance.
    """

    def __init__(
        self,
        storage: ContinuousBytesStorage,
        name: str,
        field: BytesField,
    ):
        self._name = name
        self._s = storage
        self._f = field

    def decode(self) -> npt.NDArray[np.int_ | np.float_]:
        """
        Decode field content.

        Returns
        -------
        NDArray
            decoded content.
        """
        return self._f.decode(self.content)

    @property
    def content(self) -> bytes:
        """
        Returns
        -------
        bytes
            field content.
        """
        return self._s.content[self._f.slice]

    @property
    def fld(self) -> BytesField:
        """
        Returns
        -------
        BytesField
            field instance.
        """
        return self._f

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
        return len(self) // self._f.word_size

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


class ContinuousBytesStorage:
    """
    Represents continuous storage where data storage in bytes.

    Continuous means that the storage fields must go continuously, that is,
    where one field ends, another must begin.

    Parameters
    ----------
    name: str
        name of storage.
    **fields: BytesField
        fields of the storage. The kwarg Key is used as the field name.
    """

    def __init__(self, name: str, **fields: BytesField):
        for f_name, field in fields.items():
            if not isinstance(field, BytesField):
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
            new = content[parser.fld.slice]
            if len(new):
                fields[parser.name] = new
        self.set(**fields)
        return self

    def set(self, **fields: npt.ArrayLike) -> Self:
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
                self[field].fld.infinite or len(self[field]) != 0
            ):
                continue
            diff.add(field)

        if len(diff) != 0:
            raise AttributeError(
                "missing or extra fields were found: %r" % sorted(diff)
            )

        self._set(fields)
        return self

    def _set(self, fields: dict[str, npt.ArrayLike]) -> None:
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
        parser: BytesFieldParser,
        content: npt.ArrayLike,
    ) -> bytes:
        """
        Set new content to the field.

        Parameters
        ----------
        parser: BytesFieldParser
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
            new_content = parser.fld.encode(content)

        if not parser.fld.validate(new_content):
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

    def __getitem__(self, field: str) -> BytesFieldParser:
        """Get field parser."""
        return BytesFieldParser(self, field, self._f[field])

    def __iter__(self) -> Generator[BytesFieldParser, None, None]:
        """Iterate by field parsers."""
        for field in self._f:
            yield self[field]


# todo: typehint - Generic. Create via generic for children.
class BytesFieldPattern:
    """
    Represents class which storage common parameters for field.

    Parameters
    ----------
    field_type: str, default=''
        name of field type.
    **parameters: Any
        common parameters.
    """

    _FIELD_TYPES: dict[str, type[BytesField]] = {}
    """dictionary of field types where key is a name of type."""

    def __init__(self, field_type: str = "", **parameters: Any):
        self._field_type = field_type
        self._kw = parameters

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

    def get(self, **parameters: Any) -> BytesField:
        """
        Get field initialized with parameters from pattern and from
        `parameters`.

        Parameters
        ----------
        **parameters: Any
            additional field initialization parameters.

        Returns
        -------
        BytesField
            initialized field.
        """
        # todo: check intersection and then call .get_updated
        return self._get_field_class()(**self._kw, **parameters)

    def get_updated(self, **parameters: Any) -> BytesField:
        """
        Get field initialized with parameters from pattern and from
        `parameters`.

        If parameters from pattern will be updated via `parameters` before
        creation Field instance.

        Parameters
        ----------
        **parameters: Any
            parameters for field.

        Returns
        -------
        BytesField
            initialized field.
        """
        kw_ = self._kw.copy()
        kw_.update(parameters)
        return self._get_field_class()(**kw_)  # pylint: disable=missing-kwoa

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

    def _get_field_class(self) -> type[BytesField]:
        """
        Get field class by `field_type`.

        Returns
        -------
        type[BytesField]
            field class.
        """
        return self._FIELD_TYPES.get(self._field_type, BytesField)

    @property
    def field_type(self) -> str:
        """
        Returns
        -------
        str
            name of field type.
        """
        return self._field_type

    def __init_kwargs__(self) -> dict[str, Any]:
        kwargs = {"field_type": self._field_type}
        kwargs.update(self._kw)
        return kwargs

    def __contains__(self, item: str) -> bool:
        return item in self._kw

    def __getitem__(self, parameter: str) -> Any:
        return self._kw[parameter]

    def __setitem__(self, parameter: str, value: Any) -> None:
        if parameter not in self:
            raise KeyError("%r not in parameters" % parameter)
        self._kw[parameter] = value


class BytesStoragePattern:
    """
    Represents class which storage common parameters for storage.

    Parameters
    ----------
    name: str
        name of storage.
    **kwargs: Any
        parameters for storage initialization.
    """

    def __init__(self, name: str, **kwargs: Any):
        self._name = name
        self._kw = dict(name=name, **kwargs)
        self._continuous = True  # now available only with this option
        self._p: dict[str, BytesFieldPattern] = {}

    def configure(self, **patterns: BytesFieldPattern) -> Self:
        """
        Configure storage pattern with field patterns.

        Parameters
        ----------
        **patterns : BytesFieldPattern
            dictionary of field patterns where key is a name of field.

        Returns
        -------
        Self
            self instance.
        """
        self._p = patterns
        return self

    def get(self, **update: Any) -> ContinuousBytesStorage:
        """
        Get initialized storage.

        Parameters
        ----------
        **update : Any
            those keys that are separated by "__" will be defined as
            parameters for the field, otherwise for the storage.

        Returns
        -------
        ContinuousBytesStorage
            initialized storage.

        Raises
        ------
        AssertionError
            if in some reason `_continuous` is False.
        NotConfiguredYet
            if patterns list is empty.
        """
        if len(self._p) == 0:
            raise NotConfiguredYet(self)

        for_fields, for_storage = split_complex_dict(
            update, without_sep="other"
        )
        if self._continuous:
            return self._get_continuous(for_storage, for_fields)
        raise AssertionError("not continuous storage not supports")

    def to_config(self, path: Path) -> None:
        """
        Write pattern configuration to config file.

        Parameters
        ----------
        path : Path
            path to config file.
        """
        pars = {self._name: self.__init_kwargs__()}
        for name, pattern in self:
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

        return ContinuousBytesStorage(
            **storage_kw, **fields  # type: ignore[arg-type]
        )

    def _get_fields_after_inf(
        self,
        fields_kw: dict[str, dict[str, Any]],
        inf: str,
    ) -> tuple[dict[str, BytesField], str]:
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
        tuple[dict[str, BytesField], str]
            fields - dictionary of fields from infinite (not included);
            next - name of next field after infinite.
        """
        rev_names = list(self._p)[::-1]
        fields, start, next_ = [], 0, ""
        for name in rev_names:
            if name == inf:
                break

            pattern = self._p[name]
            start -= pattern["expected"] * struct.calcsize(
                BytesField.VALUES[pattern["fmt"]]
            )
            field_kw = fields_kw[name] if name in fields_kw else {}
            field_kw.update(start=start)
            fields.append((name, pattern.get_updated(**field_kw)))
            next_ = name

        return dict(fields[::-1]), next_

    def _get_fields_before_inf(
        self, fields_kw: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, BytesField], str]:
        """
        Get the dictionary of fields that go up to and including the infinite
        field.

        Parameters
        ----------
        fields_kw : dict[str, dict[str, Any]]
            dictionary of kwargs for fields.

        Returns
        -------
        tuple[dict[str, BytesField], str]
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
            fields[name] = pattern.get_updated(**field_kw)

            stop = fields[name].stop
            if stop is None:
                inf = name
                break
            start = stop

        return fields, inf

    @classmethod
    def from_config(cls, path: Path, name: str) -> Self:
        """
        Initialize class instance by config file.

        Parameters
        ----------
        path : Path
            path to config file.
        name : str
            section with parameters for class. Use as pattern name.

        Returns
        -------
        Self
            initialized class instance.
        """
        with RWConfig(path) as cfg:
            opts = cfg.api.options(name)
            opts.pop(opts.index(name))
            return cls(**cfg.get(name, name)).configure(
                **{f: BytesFieldPattern(**cfg.get(name, f)) for f in opts}
            )

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            name of pattern.
        """
        # todo: TypedDict for this case
        return self._name

    def __init_kwargs__(self) -> dict[str, Any]:
        return self._kw

    def __getitem__(self, name: str) -> BytesFieldPattern:
        return self._p[name]

    def __iter__(
        self,
    ) -> Generator[tuple[str, BytesFieldPattern], None, None]:
        for name, pattern in self._p.items():
            yield name, pattern

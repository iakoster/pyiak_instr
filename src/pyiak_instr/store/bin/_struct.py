"""Private module of ``pyiak_instr.types.store`` with types for store
module."""
from __future__ import annotations
from abc import ABC
from typing import (
    Any,
    Generator,
    Generic,
    TypeVar,
    cast,
    overload,
)

from ...core import Code
from ...exceptions import ContentError
from ...codecs.bin import get_bytes_codec


__all__ = [
    "Field",
    "Struct",
]


# todo: drop to default
# todo: shortcut decode default?
class Field(ABC):
    """
    Represents base class for field structure.

    Parameters
    ----------
    name : str, default='std'
        field name.
    start : int, default=0
        start byte index of the field.
    stop : int, default=None
        stop byte index of the field.
    bytes_expected : int, default=0
        expected bytes count for field.
    fmt : Code, default=U8
        format for packing or unpacking the content.
    order : Code, default=BIG_ENDIAN
        bytes order for packing and unpacking.
    default : bytes, default=b''
        default value of the field.
    """

    def __init__(
        self,
        name: str = "std",
        start: int = 0,
        stop: int = None,
        bytes_expected: int = 0,
        fmt: Code = Code.U8,
        order: Code = Code.BIG_ENDIAN,
        default: bytes = b"",  # todo: to ContentType
    ) -> None:
        if len(name) == 0:
            raise ValueError("empty 'name' not allowed")
        self._validate_bytes_range(start, stop, bytes_expected)

        if bytes_expected > 0:
            stop = start + bytes_expected
            if stop == 0:
                stop = None

        self._name = name
        self._slc = slice(start, stop)
        self._fmt = fmt
        self._order = order
        self._default = default
        # todo: variant with an already initialized instance
        self._codec = get_bytes_codec(fmt, order)

        self._verify_initialized_field()

    def decode(self, content: bytes, verify: bool = False) -> Any:
        """
        Decode content from bytes with parameters from struct.

        Parameters
        ----------
        content : bytes
            content for decoding.
        verify : bool, default=False
            verify `content` before decoding.

        Returns
        -------
        Any
            decoded content.
        """
        if verify:
            self.verify(content, raise_if_false=True)
        return self._codec.decode(content)

    def encode(self, content: Any, verify: bool = False) -> bytes:
        """
        Encode content to bytes with parameters from struct.

        Parameters
        ----------
        content : Any
            content for encoding.
        verify : bool, default=False
            verify content after encoding.

        Returns
        -------
        bytes
            encoded content.
        """
        encoded = self._codec.encode(content)
        if verify:
            self.verify(encoded, raise_if_false=True)
        return encoded

    def extract(self, content: bytes) -> bytes:
        """
        Extract field content from `content` by field slice.

        Parameters
        ----------
        content : bytes
            content.

        Returns
        -------
        bytes
            field content.
        """
        return content[self.slice_]

    def verify(
        self,
        content: bytes,
        raise_if_false: bool = False,
    ) -> Code:
        """
        Verify that `content` is correct for the given field structure.

        Parameters
        ----------
        content : bytes
            content to verifying.
        raise_if_false : bool, default=False
            raise `ContentError` if content not correct.

        Returns
        -------
        Code
            OK - content is correct, other - is not.

        Raises
        ------
        ContentError
            if `raise_if_false` is True and content is not correct.
        """
        code = Code.OK

        if self.is_dynamic:
            if len(content) % self.fmt_bytesize != 0:
                code = Code.INVALID_LENGTH
        elif len(content) != self.bytes_expected:
            code = Code.INVALID_LENGTH

        if code is not Code.OK and raise_if_false:
            raise ContentError(self, clarification=repr(Code.INVALID_LENGTH))
        return code

    def _verify_initialized_field(self) -> None:
        """
        Verify values after modifying.

        Raises
        ------
        ValueError
            * if `bytes_expected` is not evenly divisible by `word_bytesize`;
            * if `default` is not correct for this struct.
        TypeError
            * if `default` specified as 1 byte in dynamic field.
        """
        if self.bytes_expected % self.fmt_bytesize:
            raise ValueError(
                "'bytes_expected' does not match an integer word count"
            )

        if (
            len(self._default) > 1
            and self.verify(self._default) is not Code.OK
        ):
            raise ValueError("default value is incorrect")

    @staticmethod
    def _validate_bytes_range(
        start: int,
        stop: int | None,
        bytes_expected: int,
    ) -> None:
        """
        Validate `start`, `stop` and `bytes_expected` values.

        Parameters
        ----------
        start : int
            start byte index of the field.
        stop : int | None
            stop byte index of the field.
        bytes_expected : int
            expected bytes count for field.

        Raises
        ------
        ValueError
            * if `bytes_expected` is a negative number;
            * if `stop` is equal to zero;
            * if `start` is negative and more than `bytes_expected`.
        TypeError
            * if `stop` and `bytes_expected` is specified.
        """
        if bytes_expected < 0:
            raise ValueError("'bytes_expected' can't be a negative number")
        if stop is not None and bytes_expected > 0:
            raise TypeError("'bytes_expected' and 'stop' setting not allowed")
        if stop == 0:
            raise ValueError("'stop' can't be equal to zero")
        if 0 > start > -bytes_expected:
            raise ValueError("it will be out of bounds")

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            field name.
        """
        return self._name

    @property
    def start(self) -> int:
        """
        Returns
        -------
        int
            start byte of the field.
        """
        return cast(int, self._slc.start)

    @property
    def stop(self) -> int | None:
        """
        Returns
        -------
        int | None
            stop byte of the field.
        """
        return cast(int | None, self._slc.stop)

    @property
    def bytes_expected(self) -> int:
        """
        Returns
        -------
        int
            expected bytes count for field. If equal to zero - field
            is dynamic.
        """
        # pylint: disable=invalid-unary-operand-type,chained-comparison
        if self.stop is None:
            if self.start < 0:
                return -self.start
            return 0
        if self.start >= 0 and self.stop < 0:
            return 0
        return self.stop - self.start

    @property
    def fmt(self) -> Code:
        """
        Returns
        -------
        Code
            format for packing or unpacking the content.
        """
        return self._fmt

    @property
    def order(self) -> Code:
        """
        Returns
        -------
        Code
            bytes order for packing and unpacking.
        """
        return self._order

    @property
    def default(self) -> bytes:
        """
        Returns
        -------
        bytes
            default value of the field. If there is one byte - it will be
            used as fill value for bytes with length as `bytes_expected`.
        """
        if len(self._default) == 1 and not self.is_dynamic:
            return self._default * self.bytes_expected
        return self._default

    @property
    def has_default(self) -> bool:
        """
        Returns
        -------
        bool
            True - default length more than zero.
        """
        return len(self._default) > 0

    @property
    def is_dynamic(self) -> bool:
        """
        Returns
        -------
        bool
            if True - field is dynamic (from empty to any).
        """
        return self.bytes_expected == 0

    @property
    def slice_(self) -> slice:
        """
        Returns
        -------
        slice
            slice with start and stop indexes of field.
        """
        return self._slc

    @property
    def fmt_bytesize(self) -> int:
        """
        Returns
        -------
        int
            count of bytes in one word.
        """
        return self._codec.fmt_bytesize

    @property
    def words_expected(self) -> int:
        """
        Returns
        -------
        int
            expected words count in the field. Returns 0 if field is infinite.
        """
        return self.bytes_expected // self.fmt_bytesize


FieldT = TypeVar("FieldT", bound=Field)


class Struct(ABC, Generic[FieldT]):
    """
    Represents base class for storage structure.

    Parameters
    ----------
    name : str, default='std'
        name of storage configuration.
    fields : dict[str, FieldT], default=None
        dictionary with fields.
    """

    def __init__(
        self,
        name: str = "std",
        fields: dict[str, FieldT] = None,
    ) -> None:
        if fields is None or len(fields) == 0:
            raise ValueError(
                f"{self.__class__.__name__} without fields not allowed"
            )
        if "" in fields:
            raise KeyError("empty field name not allowed")

        self._dyn_field = ""
        for f_name, struct in fields.items():
            s_name = struct.name
            if f_name != s_name:
                raise KeyError(
                    f"invalid struct name: {f_name!r} != {s_name!r}"
                )
            if struct.is_dynamic and self.is_dynamic:
                raise TypeError("two dynamic fields not allowed")
            if struct.is_dynamic:
                self._dyn_field = f_name

        self._name = name
        self._f = fields

    def change(
        self,
        content: bytearray,
        name: str,
        field_content: bytes,
        verify: bool = True,
    ) -> None:
        """
        Change field content in `content`.

        Parameters
        ----------
        content : bytearray
            full content.
        name : str
            field name.
        field_content : bytes
            new field content.
        verify : bool, default=True
            verify `content` and `field_content`.
        """
        field = self._f[name]
        if verify:
            self.verify(content, raise_if_false=True)
            field.verify(field_content, raise_if_false=True)
        content[field.slice_] = field_content

    @overload
    def decode(self, name: str, content: bytes) -> Any:
        ...

    @overload
    def decode(self, content: bytes) -> dict[str, Any]:
        ...

    def decode(
        self, *args: str | bytes, **kwargs: Any
    ) -> Any | dict[str, Any]:
        """
        Decode bytes content.

        Parameters
        ----------
        *args : str | bytes
            arguments for decode method (see overload).
        **kwargs : Any
            keyword arguments for decode method.

        Returns
        -------
        Any | dict[str, Any]
            decoded content.

        Raises
        ------
        TypeError
            if kwargs taken;
            if arguments is invalid.
        """
        if len(kwargs):
            raise TypeError("takes no keyword arguments")

        match args:
            case (str() as name, bytes() as content):
                return self[name].decode(content, verify=True)

            case (bytes() as content,):
                return {
                    f.name: f.decode(content[f.slice_], verify=True)
                    for f in self
                }

            case _:
                raise TypeError("invalid arguments")

    @overload
    def encode(self, content: bytes) -> dict[str, bytes]:
        ...

    @overload
    def encode(
        self, all_fields: bool = False, **fields: Any
    ) -> dict[str, bytes]:
        ...

    @overload
    def encode(
        self,
        *args: bytes,
        all_fields: bool = False,
        **kwargs: Any,
    ) -> dict[str, bytes]:
        ...

    def encode(  # type: ignore[misc]
        self,
        *args: bytes,
        all_fields: bool = False,
        **kwargs: Any,
    ) -> dict[str, bytes]:
        """
        Encode content for storage.

        Parameters
        ----------
        *args : bytes
            arguments for encode method (see overload).
        all_fields :
            check that all fields required.
        **kwargs : Any
            fields content where key is the field name.

        Returns
        -------
        dict[str, bytes]
            encoded content.

        Raises
        ------
        TypeError
            if takes args and kwargs;
            if no args or kwargs are taken.
        """
        if len(args) != 0 and len(kwargs) != 0:
            raise TypeError("takes a bytes or fields (both given)")

        if len(args) > 0:
            if len(args) != 1:
                raise TypeError(f"invalid arguments count (got {len(args)})")
            (content,) = args
            self.verify(content, raise_if_false=True)
            return {f.name: f.encode(f.extract(content)) for f in self}

        if all_fields:
            return self._get_all_fields(kwargs)
        return {f: self[f].encode(c, verify=True) for f, c in kwargs.items()}

    @overload
    def extract(
        self, content: bytes, verify: bool = True
    ) -> dict[str, bytes]:
        ...

    @overload
    def extract(  # type: ignore[misc]
        self, content: bytes, name: str, verify: bool = True
    ) -> bytes:
        ...

    @overload
    def extract(
        self, content: bytes, *names: str, verify: bool = True
    ) -> dict[str, bytes]:
        ...

    def extract(  # type: ignore[misc]
        self, content: bytes, *names: str, verify: bool = True
    ) -> bytes | dict[str, bytes]:
        """
        Extract all, one or specified fields content from `content`.

        Parameters
        ----------
        content : bytes
            content from which the fields will be extracted.
        *names : str
            field names.
        verify : bool, default=False
            varify `content` before extracting.

        Returns
        -------
        bytes | dict[str, bytes]
            - bytes
            if `names` length is equal to one - content of one field.

            - dict[str, bytes]
            if `names` length is equal to zero - content if all fields;
            if `names` length more than one - content of `names` fields.
        """
        if verify:
            self.verify(content, raise_if_false=True)

        if len(names) == 0:
            return {n: f.extract(content) for n, f in self.items()}

        if len(names) == 1:
            return self._f[names[0]].extract(content)

        return {n: self._f[n].extract(content) for n in names}

    def items(self) -> Generator[tuple[str, FieldT], None, None]:
        """
        Yields
        ------
        Generator[tuple[str, FieldStructT], None, None]
            Iterator of names and parsers.
        """
        for field in self:
            yield field.name, field

    def verify(
        self,
        content: bytes,
        raise_if_false: bool = False,
        verify_fields: bool = False,
    ) -> Code:
        """
        Check that the content is correct.

        Parameters
        ----------
        content : bytes
            content for verifying.
        raise_if_false : bool, default=False
            raise `ContentError` if content not correct.
        verify_fields : bool, default=False
            True - verify fields content.

        Returns
        -------
        Code
            OK - content is correct, other - is not.

        Raises
        ------
        ContentError
            * if content length smaller than minimal storage length;
            * if storage not dynamic and `content` too long;
            * if content for any field is not correct.
        """
        minimum_size, content_len = self.minimum_size, len(content)
        code, clarification = Code.OK, ""

        if (
            content_len < minimum_size
            or not self.is_dynamic
            and content_len > minimum_size
        ):
            code = Code.INVALID_LENGTH
            clarification = (
                f"{code!r} - expected"
                f"{' at least' if self.is_dynamic else ''} "
                f"{minimum_size}, got {content_len}"
            )

        if verify_fields and code is Code.OK:
            for field in self:
                code = field.verify(
                    field.extract(content), raise_if_false=raise_if_false
                )
                if code is not Code.OK:
                    break

        if raise_if_false and code is not Code.OK:
            raise ContentError(self, clarification=clarification)
        return code

    def _get_all_fields(self, fields: dict[str, Any]) -> dict[str, bytes]:
        """
        Encode content for all fields.

        Parameters
        ----------
        fields : dict[str, Any]
            dictionary of fields content where key is a field name.

        Returns
        -------
        dict[str, bytes]
            bytes content of all fields.

        Raises
        ------
        AssertionError
            If it is not possible to set the content of the field.
        """
        self._verify_fields_list(set(fields))
        content: dict[str, bytes] = {}

        for name, field in self.items():
            if name in fields:
                content[name] = field.encode(fields[name], verify=True)

            elif field.has_default:
                content[name] = field.default

            elif field.is_dynamic:
                content[name] = b""

            else:
                raise AssertionError(
                    f"it is impossible to encode the value for {name!r}"
                )

        return content

    def _verify_fields_list(self, fields: set[str]) -> None:
        """
        Check that fields names is correct.

        Parameters
        ----------
        fields : set[str]
            set of field names for setting.

        Raises
        ------
        AttributeError
            if extra or missing field names founded.
        """
        diff = set(self._f).symmetric_difference(fields)
        for name in diff.copy():
            if name in self:
                field = self[name]
                if field.has_default or field.is_dynamic:
                    diff.remove(name)

        if len(diff) != 0:
            raise AttributeError(
                "missing or extra fields were found: "
                f"{', '.join(map(repr, sorted(diff)))}"
            )

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            name of storage configuration.
        """
        return self._name

    @property
    def dynamic_field_name(self) -> str:
        """
        Returns
        -------
        str
            dynamic field name.
        """
        return self._dyn_field

    @property
    def is_dynamic(self) -> bool:
        """
        Returns
        -------
        bool
            True - has dynamic field.
        """
        return len(self.dynamic_field_name) != 0

    @property
    def minimum_size(self) -> int:
        """
        Returns
        -------
        int
            minimum message size in bytes.
        """
        return sum(s.bytes_expected for s in self._f.values())

    def __contains__(self, name: str) -> bool:
        """Check that field name in message."""
        return name in self._f

    def __getitem__(self, name: str) -> FieldT:
        """Get field struct."""
        return self._f[name]

    def __iter__(self) -> Generator[FieldT, None, None]:
        """Iterate by field structs."""
        for struct in self._f.values():
            yield struct

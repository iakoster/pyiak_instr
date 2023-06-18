"""Private module of ``pyiak_instr.types.store`` with types for store
module."""
from typing import (
    TYPE_CHECKING,
    Generic,
    Self,
    TypeVar,
    cast,
    overload,
)

from ...typing import WithBaseStringMethods
from ...encoders import BytesDecodeT, BytesEncodeT
from ._struct import (
    Field,
    Struct,
)

if TYPE_CHECKING:
    from typing import Any

    from ._pattern import ContainerPattern


__all__ = ["Container"]

FieldT = TypeVar("FieldT", bound=Field)
StructT = TypeVar("StructT", bound=Struct[Field])
ContainerPatternT = TypeVar(
    "ContainerPatternT", bound="ContainerPattern[Any, Any]"
)


# todo: verify current content
# todo: __format__
class Container(
    WithBaseStringMethods,
    Generic[FieldT, StructT, ContainerPatternT],
):
    """
    Represents abstract class for bytes storage.

    Parameters
    ----------
    struct : StructT
        struct instance.
    """

    def __init__(
        self,
        struct: StructT,
        pattern: ContainerPatternT | None = None,
    ) -> None:
        self._s = struct
        self._p = pattern
        self._c = bytearray()

    @overload
    def decode(self, field: str) -> BytesDecodeT:
        ...

    @overload
    def decode(self) -> dict[str, BytesDecodeT]:
        ...

    def decode(  # type: ignore[misc]
        self, *args: str
    ) -> BytesDecodeT | dict[str, BytesDecodeT]:
        """
        Decode content.

        Parameters
        ----------
        *args : str
            arguments for method.

        Returns
        -------
        BytesDecodeT | dict[str, BytesDecodeT]
            decoded content.

        Raises
        ------
        TypeError
            if arguments is invalid.
        """
        match args:
            case _ if len(args) == 0:
                return self._s.decode(self.content())

            case (str() as name,):
                return self._s.decode(name, self.content(name))

        raise TypeError("invalid arguments")

    @overload
    def encode(self, content: bytes) -> Self:
        ...

    @overload
    def encode(self, **fields: BytesEncodeT) -> Self:
        ...

    def encode(  # type: ignore[misc]
        self, *args: bytes, **kwargs: BytesEncodeT
    ) -> Self:
        """
        Encode content.

        Parameters
        ----------
        *args : bytes
            arguments for method.
        **kwargs: BytesEncodeT
            keyword arguments for method.

        Returns
        -------
        Self
            self instance.
        """
        is_empty = self.is_empty()
        encoded = self._s.encode(*args, all_fields=is_empty, **kwargs)

        if is_empty:
            for content in encoded.values():
                self._c += content

        else:
            for name, content in encoded.items():
                self._change_field_content(name, content)

        return self

    # kinda properties

    def bytes_count(self, field: str | None = None) -> int:
        """
        Get bytes count of one field or all storage (if `field` is empty).

        Parameters
        ----------
        field: str | None, default=None
            field name.

        Returns
        -------
        int
            bytes count of field or storage.
        """
        return len(self.content(field))

    def content(self, field: str | None = None) -> bytes:
        """
        Get content of one field or all storage (if `field` is empty).

        Parameters
        ----------
        field: str | None, default=None
            field name.

        Returns
        -------
        bytes
            content of field or storage.
        """
        content = self._c if field is None else self._c[self._s[field].slice_]
        return bytes(content)

    def is_empty(self, field: str | None = None) -> bool:
        """
        Get indicator that field or all storage (if `field` is None) is empty.

        Parameters
        ----------
        field : str | None, default=None
            field name.

        Returns
        -------
        bool
            indicator that field or all storage is empty.
        """
        return self.bytes_count(field) == 0

    @overload
    def words_count(self) -> dict[str, int]:
        ...

    @overload
    def words_count(self, field: str) -> int:
        ...

    def words_count(  # type: ignore[misc]
        self, *args: str
    ) -> int | dict[str, int]:
        """
        Get count of words.

        Parameters
        ----------
        *args : str
            arguments for method. allowed configuration:
                * empty - returns int;
                * (field name,) - returns dict[str, int]

        Returns
        -------
        int | dict[str, int]
            int - count of words in one specific field;
            dict[str, int] - count of words in each field.

        Raises
        ------
        TypeError
            if arguments is invalid.
        """
        match args:
            case _ if len(args) == 0:
                return {
                    n: self.bytes_count(n) // f.word_bytesize
                    for n, f in self._s.items()
                }

            case (str() as name,):
                return self.bytes_count(name) // self._s[name].word_bytesize

        raise TypeError("invalid arguments")

    def _change_field_content(self, name: str, content: bytes) -> None:
        """
        Change content of single field name.

        Parameters
        ----------
        name : str
            field name.
        content : bytes
            field content.

        Raises
        ------
        AssertionError
            if message has empty content.
        """
        assert not self.is_empty(), "message is empty"
        self._c[self._s[name].slice_] = content

    @property
    def has_pattern(self) -> bool:
        """
        Returns
        -------
        bool
            if True - storage has parent pattern.
        """
        return self._p is not None

    @property
    def pattern(self) -> ContainerPatternT:
        """
        Returns
        -------
        StoragePatternT
            parent pattern of self instance.

        Raises
        ------
        AttributeError
            if parent pattern is None (not set).
        """
        if not self.has_pattern:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no parent pattern"
            )
        return cast(ContainerPatternT, self._p)

    @property
    def struct(self) -> StructT:
        """
        Returns
        -------
        StorageStructT
            storage struct.
        """
        return self._s

    def __str_field_content__(self, name: str) -> str:
        field = self._s[name]
        content = self.content(name)
        length = len(content)
        if length == 0:
            return "EMPTY"

        step = field.word_bytesize
        if length > 20 and self.words_count(name) > 2:
            if step == 1:
                border = 4
            elif step == 2:
                border = 6
            elif step in {3, 4}:
                border = 2 * step
            else:
                border = step
        else:
            border = length

        string, start = "", 0
        while start < length:
            if start == border:
                start = length - border
                string += "... "

            stop = start + step
            word = content[start:stop].hex().lstrip("0")
            string += (word.upper() if len(word) else "0") + (
                " " if stop != length else ""
            )
            start = stop

        return string

    def __str_under_brackets__(self) -> str:
        if self.is_empty():
            return "EMPTY"
        return ", ".join(
            f"{n}={self.__str_field_content__(n)}" for n, _ in self._s.items()
        )

    def __bytes__(self) -> bytes:
        return self.content()

    def __len__(self) -> int:
        """Bytes count in message"""
        return self.bytes_count()

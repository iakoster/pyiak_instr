import unittest

from src.pyiak_instr.core import Code

from .....utils import validate_object
from .ti import TIFieldStructPattern, TIStorageStructPattern, TIStoragePattern


class TestBytesFieldStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIFieldStructPattern(typename="basic", bytes_expected=0),
            typename="basic",
            is_dynamic=True,
            size=0,
        )

    def test_get(self) -> None:
        validate_object(
            self,
            TIFieldStructPattern(
                typename="basic", bytes_expected=4, fmt=Code.U16
            ).get(),
            has_default=False,
            name="",
            stop=4,
            start=0,
            words_expected=2,
            order=Code.BIG_ENDIAN,
            word_bytesize=2,
            bytes_expected=4,
            is_dynamic=False,
            default=b"",
            slice_=slice(0, 4),
            fmt=Code.U16,
            wo_attrs=["encoder"],
        )


class TestBytesStorageStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStorageStructPattern(typename="basic", name="test"),
            typename="basic",
            name="test",
        )

    def test_get(self) -> None:
        storage = TIStorageStructPattern(
            typename="basic", name="test"
        ).configure(
                f0=TIFieldStructPattern(typename="basic", bytes_expected=3)
            ).get()

        validate_object(
            self,
            storage,
            dynamic_field_name="",
            is_dynamic=False,
            minimum_size=3,
            name="test",
            fields={},
        )
        validate_object(
            self,
            storage["f0"],
            has_default=False,
            name="f0",
            stop=3,
            start=0,
            words_expected=3,
            order=Code.BIG_ENDIAN,
            word_bytesize=1,
            bytes_expected=3,
            is_dynamic=False,
            default=b"",
            slice_=slice(0, 3),
            fmt=Code.U8,
            wo_attrs=["encoder"],
        )


class TestBytesStoragePatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStoragePattern(typename="basic", name="std"),
            typename="basic",
            name="std",
        )

    def test_get(self) -> None:
        storage_struct = TIStorageStructPattern(
            typename="basic", name="test"
        ).configure(
            f0=TIFieldStructPattern(typename="basic", bytes_expected=3)
        )
        storage = TIStoragePattern(typename="basic", name="std").configure(
                test=storage_struct
            )
        validate_object(
            self,
            storage.get(),
            typename="basic",
            name="std",
        )

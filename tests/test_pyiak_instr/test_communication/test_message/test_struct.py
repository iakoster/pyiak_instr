import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.communication.message import (
    MessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DataLengthMessageFieldStruct,
    IdMessageFieldStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
    MessageStruct,
)

from ....utils import validate_object


class TestMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            MessageFieldStruct(),
            bytes_expected=0,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=True,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, None),
            start=0,
            stop=None,
            word_bytesize=1,
            words_expected=0,
            name="",
            is_single=False,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestStaticMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            StaticMessageFieldStruct(default=b"\x00"),
            bytes_expected=1,
            default=b"\x00",
            fmt=Code.U8,
            has_default=True,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            word_bytesize=1,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestAddressMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            AddressMessageFieldStruct(),
            bytes_expected=1,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            word_bytesize=1,
            words_expected=1,
            behaviour=Code.DMA,
            units=Code.WORDS,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestCrcMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            CrcMessageFieldStruct(fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            word_bytesize=2,
            words_expected=1,
            name="",
            wo_fields=set(),
            poly=0x1021,
            init=0,
            is_single=True,
            fill_value=b"\x00",
            has_fill_value=True,
            wo_attrs=["encoder"],
        )


class TestDataMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            DataMessageFieldStruct(),
            bytes_expected=0,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=True,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, None),
            start=0,
            stop=None,
            word_bytesize=1,
            words_expected=0,
            name="",
            is_single=False,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestDataLengthMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            DataLengthMessageFieldStruct(),
            bytes_expected=1,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            word_bytesize=1,
            words_expected=1,
            name="",
            additive=0,
            behaviour=Code.ACTUAL,
            units=Code.BYTES,
            is_single=True,
            fill_value=b"\x00",
            has_fill_value=True,
            wo_attrs=["encoder"],
        )


class TestIdMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            IdMessageFieldStruct(),
            bytes_expected=1,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            word_bytesize=1,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestOperationMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            OperationMessageFieldStruct(),
            bytes_expected=1,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            word_bytesize=1,
            words_expected=1,
            descs={0: Code.READ, 1: Code.WRITE},
            descs_r={Code.READ: 0, Code.WRITE: 1},
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestResponseMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            ResponseMessageFieldStruct(),
            bytes_expected=1,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            word_bytesize=1,
            words_expected=1,
            descs={},
            descs_r={},
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestMessageStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            MessageStruct(fields={"f0": MessageFieldStruct(name="f0")}),
            minimum_size=0,
            dynamic_field_name="f0",
            mtu=1500,
            is_dynamic=True,
            name="std",
            divisible=False,
            wo_attrs=["get", "has", "fields"]
        )

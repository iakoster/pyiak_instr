import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotAmongTheOptions
from src.pyiak_instr.communication.message import (
    MessageFieldStruct,
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DataLengthMessageFieldStruct,
    IdMessageFieldStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
)

from ....utils import validate_object


class TestMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            MessageFieldStruct(
                start=10, fmt=Code.I64, bytes_expected=160
            ),
            bytes_expected=160,
            default=b"",
            fmt=Code.I64,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(10, 170),
            start=10,
            stop=170,
            word_bytesize=8,
            words_expected=20,
        )


class TestSingleMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            SingleMessageFieldStruct(start=10, fmt=Code.I64),
            bytes_expected=8,
            default=b"",
            fmt=Code.I64,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(10, 18),
            start=10,
            stop=18,
            word_bytesize=8,
            words_expected=1,
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            SingleMessageFieldStruct(start=0, fmt=Code.U8, stop=2)
        self.assertEqual(
            "single field should expect one word", exc.exception.args[0]
        )


class TestStaticMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            StaticMessageFieldStruct(start=0, fmt=Code.U8, default=b"\x00"),
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
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            StaticMessageFieldStruct(start=0, fmt=Code.U8)
        self.assertEqual("default value not specified", exc.exception.args[0])

    def test_verify(self) -> None:
        for i, (data, ref) in enumerate(((b"2", False), (b"4", True))):
            with self.subTest(test=i):
                self.assertEqual(ref, StaticMessageFieldStruct(
                    start=0, fmt=Code.U8, default=b"4"
                ).verify(data))


class TestAddressMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            AddressMessageFieldStruct(start=0, fmt=Code.U8),
            behaviour=Code.DMA,
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
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(NotAmongTheOptions) as exc:
            AddressMessageFieldStruct(
                start=0, fmt=Code.U8, behaviour=Code.EXPECTED
            )
        self.assertEqual(
            "behaviour option not in {<Code.DMA: 1539>, <Code.STRONG: 1540>}"
            ", got <Code.EXPECTED: 1541>",
            exc.exception.args[0],
        )


class TestCrcMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            CrcMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            init=0,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            poly=0x1021,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            word_bytesize=2,
            words_expected=1,
        )

    def test_get_crc(self) -> None:
        contents = (
            b"\xaa",
            b"\xaa\xbb",
            b"\xff\xaa\x88\x44",
            b"\xff\xaa\x88\x44\xff\xaa\x88\x44",
            b"\x00\x10\x20\x30\x40\x50\x60\x70\x80\x90\xa0\xb0\xc0\xd0\xe0"
            b"\xf0",
        )
        data = {
            "CRC-16/XMODEM": dict(
                fmt=Code.U16,
                init=0,
                poly=0x1021,
                crc=(0x14A0, 0xE405, 0xAB8F, 0x488D, 0x4375),
            )
        }
        for name in data:
            dict_: dict = data[name].copy()
            crcs = dict_.pop("crc")
            obj = CrcMessageFieldStruct(start=0, **dict_)

            assert len(contents) == len(crcs)
            for i, (content, crc) in enumerate(zip(contents, crcs)):
                with self.subTest(test=name, crc=i):
                    self.assertEqual(crc, obj.get_crc(content))


class TestDataMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            DataMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=0,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            is_dynamic=True,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, None),
            start=0,
            stop=None,
            word_bytesize=2,
            words_expected=0,
        )


class TestDataLengthMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            DataLengthMessageFieldStruct(start=0, fmt=Code.U16),
            additive=0,
            behaviour=Code.ACTUAL,
            bytes_expected=2,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            units=Code.BYTES,
            word_bytesize=2,
            words_expected=1,
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="negative additive"):
            with self.assertRaises(ValueError) as exc:
                DataLengthMessageFieldStruct(
                    start=0, fmt=Code.U8, additive=-1
                )
            self.assertEqual(
                "additive number must be positive integer",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid behaviour"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                DataLengthMessageFieldStruct(
                    start=0, fmt=Code.U8, behaviour=Code.DMA
                )
            self.assertEqual(
                "behaviour option not allowed, got <Code.DMA: 1539>",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid units"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                DataLengthMessageFieldStruct(
                    start=0, fmt=Code.U8, units=Code.INT
                )
            self.assertEqual(
                "units option not allowed, got <Code.INT: 261>",
                exc.exception.args[0],
            )


class TestIdMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            IdMessageFieldStruct(start=0, fmt=Code.U16),
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
        )


class TestOperationMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            OperationMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            descriptions={Code.READ: 0, Code.WRITE: 1},
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            word_bytesize=2,
            words_expected=1,
        )


class TestResponseMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            ResponseMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            default_code=Code.UNDEFINED,
            descriptions={},
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            word_bytesize=2,
            words_expected=1,
        )

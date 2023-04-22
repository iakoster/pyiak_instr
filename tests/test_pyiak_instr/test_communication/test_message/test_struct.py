import unittest
from typing import Any

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
    MessageFieldStructUnionT,
    MessageFieldPattern,
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
            wo_fields=set(),
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
            descs={0: Code.READ, 1: Code.WRITE},
            descs_r={Code.READ: 0, Code.WRITE: 1},
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
            descs={},
            descs_r={},
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


class TestMessageFieldPattern(unittest.TestCase):

    def test_get(self) -> None:
        self.assertIsInstance(
            MessageFieldPattern("basic", start=0).get(fmt=Code.U8),
            MessageFieldStruct,
        )

    def test_basic(self) -> None:
        self._validate(
            MessageFieldStruct,
            MessageFieldPattern.basic(fmt=Code.U16).get(start=10),
            0,
            slice(10, None),
            2,
        )

    def test_single(self) -> None:
        self._validate(
            SingleMessageFieldStruct,
            MessageFieldPattern.single(fmt=Code.U24).get(start=3),
            3,
            slice(3, 6),
            3,
        )

    def test_static(self) -> None:
        self._validate(
            StaticMessageFieldStruct,
            MessageFieldPattern.static(
                fmt=Code.U40, default=b"\x00\x02\x03\x04\x05"
            ).get(start=2),
            5,
            slice(2, 7),
            5,
            default=b"\x00\x02\x03\x04\x05",
        )

    def test_address(self) -> None:
        self._validate(
            AddressMessageFieldStruct,
            MessageFieldPattern.address(fmt=Code.U8).get(start=0),
            1,
            slice(0, 1),
            1,
            behaviour=Code.DMA,
        )

    def test_crc(self) -> None:
        self._validate(
            CrcMessageFieldStruct,
            MessageFieldPattern.crc(fmt=Code.U16).get(start=-2),
            2,
            slice(-2, None),
            2,
            poly=0x1021,
            init=0,
            wo_fields=set(),
        )

    def test_data(self) -> None:
        self._validate(
            DataMessageFieldStruct,
            MessageFieldPattern.data(
                fmt=Code.U8, bytes_expected=1
            ).get(start=-3),
            1,
            slice(-3, -2),
            1,
        )

    def test_data_length(self) -> None:
        self._validate(
            DataLengthMessageFieldStruct,
            MessageFieldPattern.data_length(fmt=Code.U8).get(start=4),
            1,
            slice(4, 5),
            1,
            behaviour=Code.ACTUAL,
            units=Code.BYTES,
            additive=0,
        )

    def test_id(self) -> None:
        self._validate(
            IdMessageFieldStruct,
            MessageFieldPattern.id_(fmt=Code.U48).get(start=5),
            6,
            slice(5, 11),
            6,
        )

    def test_operation(self) -> None:
        self._validate(
            OperationMessageFieldStruct,
            MessageFieldPattern.operation(fmt=Code.U16).get(start=1),
            2,
            slice(1, 3),
            2,
            descs={Code.READ: 0, Code.WRITE: 1},
            descs_r={0: Code.READ, 1: Code.WRITE},
        )

    def test_response(self) -> None:
        self._validate(
            ResponseMessageFieldStruct,
            MessageFieldPattern.response(fmt=Code.U8).get(start=0),
            1,
            slice(0, 1),
            1,
            descs={},
            descs_r={},
        )

    def _validate(
            self,
            ref_class: type[MessageFieldStructUnionT],
            ref: MessageFieldStructUnionT,
            bytes_expected: int,
            slice_: slice,
            word_bytesize: int,
            default=b"",
            order: Code = Code.BIG_ENDIAN,
            **kwargs: Any,
    ) -> None:
        self.assertIsInstance(ref, ref_class)
        kw = dict(
            bytes_expected=bytes_expected,
            default=default,
            order=order,
            slice_=slice_,
            word_bytesize=word_bytesize,
        )
        kw.update(**kwargs)

        validate_object(
            self,
            ref,
            **{k: kw[k] for k in sorted(kw)},
            wo_attrs=[
                "fmt",
                "has_default",
                "is_dynamic",
                "start",
                "stop",
                "words_expected",
            ],
        )

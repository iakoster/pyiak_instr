import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.communication.message import (
    MessageField,
    SingleMessageField,
    StaticMessageField,
    AddressMessageField,
    CrcMessageField,
    DataMessageField,
    DataLengthMessageField,
    IdMessageField,
    OperationMessageField,
    ResponseMessageField,
)

from ....utils import validate_object


class TestMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            MessageField(
                expected=20,
                fmt=Code.I64,
                start=10,
            ),
            bytes_expected=160,
            default=b"",
            expected=20,
            fmt=Code.I64,
            infinite=False,
            order=Code.BIG_ENDIAN,
            slice=slice(10, 170),
            start=10,
            stop=170,
            word_size=8,
        )


class TestSingleMessageField(unittest.TestCase):

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            SingleMessageField(
                fmt=Code.U8,
                start=0,
                expected=2,
            )
        self.assertEqual(
            "single field should expect one word", exc.exception.args[0]
        )

    def test_validate(self) -> None:
        datas = (
            (b"", False),
            (b"\x00", False),
            (b"\x00" * 2, True),
            (b"\x00" * 3, False),
        )
        obj = SingleMessageField(
            fmt=Code.U16,
            start=0,
        )

        for i, (data, ref) in enumerate(datas):
            with self.subTest(test=i):
                self.assertEqual(ref, obj.validate(data))


class TestStaticMessageField(unittest.TestCase):

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            StaticMessageField(
                fmt=Code.U8,
                start=0,
            )
        self.assertEqual("default value not specified", exc.exception.args[0])

    def test_validate(self) -> None:
        obj = StaticMessageField(
            fmt=Code.U16,
            start=0,
            default=b"42"
        )
        for i, (data, ref) in enumerate((
            (b"41", False),
            (b"42", True),
        )):
            self.assertEqual(ref, obj.validate(data))


class TestAddressMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            AddressMessageField(
                fmt=Code.I64,
                start=10,
            ),
            bytes_expected=8,
            default=b"",
            expected=1,
            fmt=Code.I64,
            infinite=False,
            order=Code.BIG_ENDIAN,
            slice=slice(10, 18),
            start=10,
            stop=18,
            word_size=8,
        )


class TestCrcMessageField(unittest.TestCase):

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            CrcMessageField(fmt=Code.U8, start=0, algorithm_name="None")
        self.assertEqual("invalid algorithm: 'None'", exc.exception.args[0])

    def test_algorithm(self) -> None:
        obj = CrcMessageField(fmt=Code.U16, start=0)

        for i, (data, ref) in enumerate((
            (b"\x10\x01\x20\x04", 0x6af5),
            (bytes(range(15)), 0x9b92),
            (bytes(i % 256 for i in range(1500)), 0x9243),
            (b"\x01\x00\x00\x00\x00\x00", 0x45a0),
        )):
            with self.subTest(test=i):
                self.assertEqual(ref, obj.algorithm(data))


class TestDataMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            DataMessageField(
                fmt=Code.I64,
                start=10,
                expected=-1,
            ),
            bytes_expected=0,
            default=b"",
            expected=0,
            fmt=Code.I64,
            infinite=True,
            order=Code.BIG_ENDIAN,
            slice=slice(10, None),
            start=10,
            stop=None,
            word_size=8,
        )


class TestDataLengthMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            DataLengthMessageField(
                fmt=Code.U16,
                start=10,
            ),
            additive=0,
            behaviour=Code.ACTUAL,
            bytes_expected=2,
            default=b"",
            expected=1,
            fmt=Code.U16,
            infinite=False,
            order=Code.BIG_ENDIAN,
            slice=slice(10, 12),
            start=10,
            stop=12,
            units=Code.BYTES,
            word_size=2,
        )

    def test_init_exc(self) -> None:

        for i, (msg, kw) in enumerate((
            (
                "additive number must be positive integer, got -1",
                {"additive": -1},
            ), (
                "invalid behaviour: <Code.WRITE: 1538>",
                {"behaviour": Code.WRITE},
            ), (
                "invalid units: <Code.ACTUAL: 1536>",
                {"units": Code.ACTUAL},
            ),
        )):
            with self.subTest(test=i):
                with self.assertRaises(ValueError) as exc:
                    DataLengthMessageField(fmt=Code.U8, start=0, **kw)
                self.assertEqual(msg, exc.exception.args[0])


class TestIdMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            IdMessageField(
                fmt=Code.U16,
                start=10,
            ),
            bytes_expected=2,
            default=b"",
            expected=1,
            fmt=Code.U16,
            infinite=False,
            order=Code.BIG_ENDIAN,
            slice=slice(10, 12),
            start=10,
            stop=12,
            word_size=2,
        )


class TestOperationMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            OperationMessageField(
                fmt=Code.U16,
                start=10,
            ),
            bytes_expected=2,
            default=b"",
            descriptions={Code.READ: 0, Code.WRITE: 1},
            expected=1,
            fmt=Code.U16,
            infinite=False,
            order=Code.BIG_ENDIAN,
            slice=slice(10, 12),
            start=10,
            stop=12,
            word_size=2,
        )


class TestResponseMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            ResponseMessageField(
                fmt=Code.U16,
                start=10,
                codes={0: Code.OK, 1: Code.ERROR},
            ),
            bytes_expected=2,
            codes={0: Code.OK, 1: Code.ERROR},
            default=b"",
            default_code=Code.UNDEFINED,
            expected=1,
            fmt=Code.U16,
            infinite=False,
            order=Code.BIG_ENDIAN,
            slice=slice(10, 12),
            start=10,
            stop=12,
            word_size=2,
        )

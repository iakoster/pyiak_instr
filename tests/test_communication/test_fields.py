import unittest

import numpy as np

from pyinstr_iakoster.communication import (
    Field,
    FloatWordsCountError,
    PartialFieldError
)


class TestField(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = Field("f", "n", 1, -1, ">H")

    def test_base_init(self):
        tf = Field(
            "format",
            "name",
            1,
            4,
            ">B",
            info={"info": True},
            content=b"\x01\x02\x03\x04"
        )
        attributes = dict(
            package_format="format",
            name="name",
            info={"info": True},
            start_byte=1,
            end_byte=5,
            expected=4,
            finite=True,
            fmt=">B",
            bytesize=1,
            content=b"\x01\x02\x03\x04",
            words_count=4,
        )

        for name, val in attributes.items():
            with self.subTest(name=name):
                self.assertEqual(val, tf.__getattribute__(name))
        with self.subTest(name="slice"):
            self.assertEqual(1, tf.slice.start)
            self.assertEqual(5, tf.slice.stop)
        with self.subTest(name="field_class"):
            self.assertIs(Field, tf.field_class)

    def test_base_init_infinite(self):
        tf = Field(
            "format", "name", 1, -1, ">B", content=b"\x01\x02\x03\x04"
        )
        attributes = dict(
            info={},
            start_byte=1,
            end_byte=np.inf,
            expected=-1,
            finite=False,
        )

        for name, val in attributes.items():
            with self.subTest(name=name):
                self.assertEqual(val, tf.__getattribute__(name))
        with self.subTest(name="slice"):
            self.assertEqual(1, tf.slice.start)
            self.assertEqual(None, tf.slice.stop)

    def test_base_magic_basic(self):
        tf = Field("format", "name", 1, 4, ">B", content=b"\x01\x02\x03\x04")
        with self.subTest(method="bytes"):
            self.assertEqual(b"\x01\x02\x03\x04", bytes(tf))
        with self.subTest(method="len"):
            self.assertEqual(4, len(tf))
        with self.subTest(method="repr"):
            self.assertEqual("<Field(1 2 3 4, fmt='>B')>", repr(tf))

    def test_base_magic_additional(self):
        tf = Field("format", "name", 0, -1, ">H")
        tf.set_content(range(20))
        with self.subTest(method="repr"):
            self.assertEqual(
                "<Field(0 1 2 3 4 5 6 7 ...(12), fmt='>H')>", repr(tf)
            )

    def test_init_exc_float(self):
        with self.assertRaises(FloatWordsCountError) as exc:
            Field(
                "format", "name", 0, 2, ">H", content=b"\x01\x02\x03"
            )
        self.assertEqual(
            "not integer count of words in the Field (expected 2, got 1.5)",
            exc.exception.args[0]
        )

    def test_init_exc_partial(self):
        with self.assertRaises(PartialFieldError) as exc:
            Field(
                "format", "name", 0, 3, ">H", content=b"\x01\x02"
            )
        self.assertEqual(
            "the Field is incomplete (filled to 0.3)",
            exc.exception.args[0]
        )

    def test_init_exc_partial_more(self):
        with self.assertRaises(PartialFieldError) as exc:
            Field(
                "format", "name", 0, 3, ">H", content=b"\x01\x02" * 5
            )
        self.assertEqual(
            "the Field is incomplete (filled to 1.7)",
            exc.exception.args[0]
        )

    def test_set_content_bytes(self):
        self.tf.set_content(b"\x01\x02\x03\x04" * 2)
        self.assertEqual(b"\x01\x02\x03\x04" * 2, self.tf.content)

    def test_set_content_bytearray(self):
        self.tf.set_content(bytearray(b"\x01\x02\x03\x04" * 2))
        self.assertEqual(b"\x01\x02\x03\x04" * 2, self.tf.content)

    def test_set_content_ndarray(self):
        data = np.arange(4, dtype=np.uint8)
        self.tf.set_content(data)
        self.assertEqual(b"\x00\x00\x00\x01\x00\x02\x00\x03", self.tf.content)

    def test_set_content_iter(self):
        self.tf.set_content(range(4))
        self.assertEqual(b"\x00\x00\x00\x01\x00\x02\x00\x03", self.tf.content)

    def test_set_content_iter_2(self):
        self.tf.set_content([i for i in range(4)])
        self.assertEqual(b"\x00\x00\x00\x01\x00\x02\x00\x03", self.tf.content)

    def test_set_content_nums(self):
        numbers = {
            b"\x00\x01": 1,
            b"\x00\x02": np.uint(2),
            b"\x00\x03": np.uint8(3),
            b"\x00\x04": np.uint64(4),
            b"\x00\x05": np.int64(5),
            b"\x00\x06": np.int8(6),
            b"\x00\x07": np.int_(7),
        }
        for i_test, (bytes_, val) in enumerate(numbers.items()):
            with self.subTest(i_test=i_test, type=type(val)):
                self.tf.set_content(val)
                self.assertEqual(bytes_, self.tf.content)

    def test_set_content_float(self):
        tf = Field("format", "name", 0, -1, ">f")
        tf.set_content(1.263)
        self.assertEqual(
            b"\x3f\xa1\xa9\xfc",
            tf.content,
        )

    def test_set_content_empty(self):
        self.tf.set_content(b"")
        self.assertEqual(b"", self.tf.content)

    def test_extract_from(self):
        tf = Field("format", "name", 2, -1, ">B")
        tf.extract_from(b"\x01\x02\x03\x04\x05\x06")
        self.assertEqual(b"\x03\x04\x05\x06", tf.content)

        tf = Field("format", "name", 2, 2, ">B")
        tf.extract_from(b"\x01\x02\x03\x04\x05\x06")
        self.assertEqual(b"\x03\x04", tf.content)

    def test_extract_from_empty(self):
        with self.assertRaises(ValueError) as exc:
            self.tf.extract_from(b"")
        self.assertEqual(
            "Unable to extract because the incoming message is empty",
            exc.exception.args[0]
        )

    def test_unpack(self):
        data = np.arange(4)
        self.tf.set_content(data)
        self.assertTrue((data == self.tf.unpack()).all())

    def test_unpack_custom(self):
        self.tf.set_content(b"\xf4\xa9\x12\x8a")
        self.assertTrue(np.isclose([-1.0716238e+32], self.tf.unpack(">f")))

    def test_hex(self):
        self.tf.set_content(range(4))
        self.assertEqual("0000 0001 0002 0003", self.tf.hex())

    def test_magic_iter(self):
        self.tf.set_content(range(4))
        self.assertListEqual([i for i in range(4)], [i for i in self.tf])

    def test_magic_getitem(self):
        self.tf.set_content(range(4))
        self.assertEqual(2, self.tf[2])
        self.assertTrue(([0, 1] == self.tf[:2]).all())

    def test_magic_str(self):
        self.tf.set_content([0x23, 0xff12, 0x521, 0x12])
        self.assertEqual("23 FF12 521 12", str(self.tf))

import unittest
from typing import Any

import numpy as np

from pyinstr_iakoster.communication import (
    Field,
    FieldSingle,
    FieldStatic,
    FieldAddress,
    FieldData,
    FieldDataLength,
    FieldOperation,
    FloatWordsCountError,
    PartialFieldError
)


def compare_fields_base(
        test_case: unittest.TestCase,
        field: Field,
        attributes: dict[str, Any],
        slice_start: int | None,
        slice_end: int | None,
        class_instance: type
):
    for name, val in attributes.items():
        with test_case.subTest(name=name):
            test_case.assertEqual(val, field.__getattribute__(name))
    with test_case.subTest(name="slice"):
        test_case.assertEqual(slice_start, field.slice.start)
        test_case.assertEqual(slice_end, field.slice.stop)
    with test_case.subTest(name="field_class"):
        test_case.assertIs(class_instance, field.field_class)


class TestField(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = Field("f", "n", 1, -1, ">H")

    def test_base_init(self):
        compare_fields_base(
            self, Field(
                "format",
                "name",
                1,
                4,
                ">B",
                info={"info": True},
                content=b"\x01\x02\x03\x04"
            ), dict(
                format_name="format",
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
            ), 1, 5, Field
        )

    def test_base_init_infinite(self):
        compare_fields_base(
            self, Field(
                "format", "name", 1, -1, ">B", content=b"\x01\x02\x03\x04"
            ), dict(
                info={},
                start_byte=1,
                end_byte=np.inf,
                expected=-1,
                finite=False,
            ), 1, None, Field
        )

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
        tf.set(range(20))
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

    def test_set_bytes(self):
        self.tf.set(b"\x01\x02\x03\x04" * 2)
        self.assertEqual(b"\x01\x02\x03\x04" * 2, self.tf.content)

    def test_set_bytearray(self):
        self.tf.set(bytearray(b"\x01\x02\x03\x04" * 2))
        self.assertEqual(b"\x01\x02\x03\x04" * 2, self.tf.content)

    def test_set_ndarray(self):
        data = np.arange(4, dtype=np.uint8)
        self.tf.set(data)
        self.assertEqual(b"\x00\x00\x00\x01\x00\x02\x00\x03", self.tf.content)

    def test_set_iter(self):
        self.tf.set(range(4))
        self.assertEqual(b"\x00\x00\x00\x01\x00\x02\x00\x03", self.tf.content)

    def test_set_iter_2(self):
        self.tf.set([i for i in range(4)])
        self.assertEqual(b"\x00\x00\x00\x01\x00\x02\x00\x03", self.tf.content)

    def test_set_nums(self):
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
                self.tf.set(val)
                self.assertEqual(bytes_, self.tf.content)

    def test_set_float(self):
        tf = Field("format", "name", 0, -1, ">f")
        tf.set(1.263)
        self.assertEqual(
            b"\x3f\xa1\xa9\xfc",
            tf.content,
        )

    def test_set_empty(self):
        self.tf.set(b"")
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
        self.tf.set(data)
        self.assertTrue((data == self.tf.unpack()).all())

    def test_unpack_custom(self):
        self.tf.set(b"\xf4\xa9\x12\x8a")
        self.assertTrue(np.isclose([-1.0716238e+32], self.tf.unpack(">f")))

    def test_hex(self):
        self.tf.set(range(4))
        self.assertEqual("0000 0001 0002 0003", self.tf.hex())

    def test_magic_iter(self):
        self.tf.set(range(4))
        self.assertListEqual([i for i in range(4)], [i for i in self.tf])

    def test_magic_getitem(self):
        self.tf.set(range(4))
        self.assertEqual(2, self.tf[2])
        self.assertTrue(([0, 1] == self.tf[:2]).all())

    def test_magic_str(self):
        self.tf.set([0x23, 0xff12, 0x521, 0x12])
        self.assertEqual("23 FF12 521 12", str(self.tf))


class TestFieldSingle(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = FieldSingle("f", "n", 0, ">H")

    def test_base_init(self):
        compare_fields_base(
            self,
            FieldSingle(
                "format",
                "name",
                1,
                ">H",
                info={"info": True},
                content=0xfa1c
            ), dict(
                format_name="format",
                name="name",
                info={"info": True},
                start_byte=1,
                end_byte=3,
                expected=1,
                finite=True,
                fmt=">H",
                bytesize=2,
                content=b"\xfa\x1c",
                words_count=1,
            ), 1, 3, FieldSingle
        )

    def test_unpack(self):
        self.assertEqual(None, self.tf.unpack())
        self.tf.set(0x123)
        self.assertEqual(0x123, self.tf.unpack())

    def test_madic_iter(self):
        self.tf.set(0xfa12)
        self.assertListEqual([0xfa12], [i for i in self.tf])

    def test_magic_getitem(self):
        with self.assertRaises(AttributeError) as exc:
            a = self.tf[0]
        self.assertEqual(
            "disallowed inherited",
            exc.exception.args[0]
        )


class TestFieldStatic(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = FieldStatic("f", "n", 0, ">H", content=0x1234)

    def test_base_init(self):
        compare_fields_base(
            self,
            FieldStatic(
                "format",
                "name",
                0,
                ">I",
                info={"info": True},
                content=0xfa1c
            ), dict(
                format_name="format",
                name="name",
                info={"info": True},
                start_byte=0,
                end_byte=4,
                expected=1,
                finite=True,
                fmt=">I",
                bytesize=4,
                content=b"\x00\x00\xfa\x1c",
                words_count=1,
            ), 0, 4, FieldStatic
        )

    def test_set(self):
        with self.assertRaises(AttributeError) as exc:
            self.tf.set(0x4321)
        self.assertEqual(
            "disallowed inherited",
            exc.exception.args[0]
        )


class TestFieldAddress(unittest.TestCase):

    def test_base_init(self):
        compare_fields_base(
            self,
            FieldAddress(
                "format",
                0,
                ">I",
                info={"info": True}
            ), dict(
                format_name="format",
                name="address",
                info={"info": True},
                start_byte=0,
                end_byte=4,
                expected=1,
                finite=True,
                fmt=">I",
                bytesize=4,
                content=b"",
                words_count=0,
            ), 0, 4, FieldAddress
        )


class TestFieldData(unittest.TestCase):

    def test_base_init(self):
        compare_fields_base(
            self,
            FieldData(
                "format",
                0,
                2,
                ">I",
                info={"info": True}
            ), dict(
                format_name="format",
                name="data",
                info={"info": True},
                start_byte=0,
                end_byte=8,
                expected=2,
                finite=True,
                fmt=">I",
                bytesize=4,
                content=b"",
                words_count=0,
            ), 0, 8, FieldData
        )


class TestFieldDataLength(unittest.TestCase):

    @staticmethod
    def get_tf(units: int, additive: int) -> FieldDataLength:
        return FieldDataLength("f", 0, ">H", units=units, additive=additive)

    @staticmethod
    def get_tf_data(content):
        return FieldData("f", 0, 2, ">H", content)

    def test_base_init(self):
        compare_fields_base(
            self,
            FieldDataLength(
                "format",
                0,
                ">H",
                additive=0,
            ), dict(
                format_name="format",
                name="data_length",
                info={},
                start_byte=0,
                end_byte=2,
                expected=1,
                finite=True,
                fmt=">H",
                bytesize=2,
                content=b"",
                words_count=0,
                units=0x10,
                additive=0,
            ), 0, 2, FieldDataLength
        )

    def test_base_init_other(self):
        compare_fields_base(
            self,
            FieldDataLength(
                "format",
                0,
                ">H",
                units=0x11,
                additive=10,
            ), dict(
                format_name="format",
                name="data_length",
                info={},
                start_byte=0,
                end_byte=2,
                expected=1,
                finite=True,
                fmt=">H",
                bytesize=2,
                content=b"",
                words_count=0,
                units=0x11,
                additive=10,
            ), 0, 2, FieldDataLength
        )

    def test_init_wrong_oper_core(self):
        with self.assertRaises(ValueError) as exc:
            FieldDataLength("f", 0, "b", units=0x12)
        self.assertEqual("invalid units: 18", exc.exception.args[0])

    def test_init_wrong_additive(self):
        with self.assertRaises(ValueError) as exc:
            FieldDataLength("f", 0, "b", additive=-1)
        self.assertEqual(
            "additive number must be integer and positive, got -1",
            exc.exception.args[0]
        )

    def test_update(self):
        tf_data = self.get_tf_data([0x12, 0x14])
        init_args = (
            (FieldDataLength.BYTES, 0),
            (FieldDataLength.BYTES, 4),
            (FieldDataLength.WORDS, 0),
            (FieldDataLength.WORDS, 7)
        )
        results = (4, 8, 2, 9)
        for i_test, (args, result) in enumerate(zip(init_args, results)):
            with self.subTest(i_test=i_test):
                tf = self.get_tf(*args)
                tf.update(tf_data)
                self.assertEqual(
                    result,
                    tf.unpack()
                )

    def test_update_invalid_units(self):
        tf = self.get_tf(0x10, 0)
        tf._units = 0
        with self.assertRaises(ValueError) as exc:
            tf.update(self.get_tf_data([0, 1]))
        self.assertEqual("invalid units: 0", exc.exception.args[0])


class TestFieldOperation(unittest.TestCase):

    @staticmethod
    def get_tf(desc_dict=None) -> FieldOperation:
        return FieldOperation("f", 0, ">H", desc_dict=desc_dict)

    def test_base_init(self):
        compare_fields_base(
            self,
            FieldOperation(
                "format",
                0,
                ">H",
                content=1
            ), dict(
                format_name="format",
                name="operation",
                info={},
                start_byte=0,
                end_byte=2,
                expected=1,
                finite=True,
                fmt=">H",
                bytesize=2,
                content=b"\x00\x01",
                words_count=1,
                base="w",
                desc="w",
                desc_dict={'r': 0, 'w': 1, 'e': 2},
                desc_dict_rev={0: 'r', 1: 'w', 2: 'e'},
            ), 0, 2, FieldOperation
        )

    def test_base_init_custom_desc_dict(self):
        tf = FieldOperation(
                "format",
                0,
                ">H",
                content="w1",
                desc_dict={"r1": 0x2, "r2": 0xf1, "w1": 0xf}
            )
        compare_fields_base(
            self, tf, dict(
                content=b"\x00\x0f",
                words_count=1,
                base="w",
                desc="w1",
                desc_dict={"r1": 0x2, "r2": 0xf1, "w1": 0x0f},
                desc_dict_rev={0x2: "r1", 0xf1: "r2", 0x0f: "w1"},
            ), 0, 2, FieldOperation
        )
        tf.set("r2")
        self.assertEqual(0xf1, tf.unpack())
        self.assertEqual("r2", tf.desc)
        tf.set(0x2)
        self.assertEqual(0x2, tf.unpack())
        self.assertEqual("r1", tf.desc)

    def test_update_desc_std(self):
        contents = (0, 1, 2)
        descs = ("r", "w", "e")
        tf = self.get_tf()
        for i_test, (content, desc) in enumerate(zip(contents, descs)):
            with self.subTest(i_test=i_test):
                tf.set(content)
                self.assertEqual(desc, tf.desc)
                self.assertEqual(content, tf.unpack())
                tf.set(desc)
                self.assertEqual(desc, tf.desc)
                self.assertEqual(content, tf.unpack())

    def test_update_desc(self):
        contents = (0, 1, 2, 0xff)
        descs = ("r1", "r2", "w1", "er")
        tf = self.get_tf({"r1": 0, "r2": 1, "w1": 2, "er": 0xff})
        for i_test, (content, desc) in enumerate(zip(contents, descs)):
            with self.subTest(i_test=i_test):
                tf.set(content)
                self.assertEqual(desc, tf.desc)
                self.assertEqual(content, tf.unpack())
                tf.set(desc)
                self.assertEqual(desc, tf.desc)
                self.assertEqual(content, tf.unpack())

    def test_compare_magic_ne(self):
        contents = (0, 2, 0xff)
        descs = ("r", "w", "e")
        tf = self.get_tf({"r1": 0, "w1": 2, "er": 0xff})
        tf1 = self.get_tf({"r1": 0, "w1": 2, "er": 0xff})

        results_map = (
            (True, False, False),
            (False, True, False),
            (False, False, True)
        )

        for i_test, tf_cont in enumerate(contents):
            tf.set(tf_cont)
            for i_subtest, tf1_cont in enumerate(contents):
                with self.subTest(i_test=i_test, i_subtest=i_subtest):
                    tf1.set(tf1_cont)
                    result = results_map[i_test][i_subtest]
                    self.assertIs(result, tf.compare(tf1))
                    self.assertIs(result, tf == tf1.base)
                    self.assertIs(not result, tf != tf1)
                    self.assertEqual(descs[i_test], tf.base)
                    self.assertEqual(descs[i_subtest], tf1.base)

    def test_compare_invalid_type(self):
        with self.assertRaises(TypeError) as exc:
            self.get_tf().compare(1)
        self.assertEqual(
            "invalid class for comparsion: <class 'int'>",
            exc.exception.args[0]
        )


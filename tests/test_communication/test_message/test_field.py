import unittest
from typing import Any, get_args

import numpy as np

from ..utils import validate_object

from pyinstr_iakoster.core import Code
from pyinstr_iakoster.communication import (
    Field,
    SingleField,
    StaticField,
    CrcField,
    AddressField,
    DataField,
    DataLengthField,
    OperationField,
    ResponseField,
    FieldType,
    FieldSetter,
    FieldMessage,
    FieldContentError,
)


class TestField(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = Field(
            "f", "n", start_byte=1, expected=-1, fmt=">H"
        )

    def test_base_init(self):
        validate_object(
            self,
            Field(
                "format",
                "name",
                start_byte=1,
                expected=4,
                fmt=">B",
            ),
            slice=slice(1, 5),
            mf_name="format",
            name="name",
            start_byte=1,
            stop_byte=5,
            expected=4,
            finite=True,
            may_be_empty=False,
            fmt=">B",
            bytesize=1,
            default=b"",
            content=b"",
            words_count=0,
            check_attrs=True,
            wo_attrs=["parent"],
        )

    def test_base_init_infinite(self):
        validate_object(
            self,
            Field(
                "format",
                "name",
                start_byte=1,
                expected=-1,
                fmt=">B",
            ),
            slice=slice(1, None),
            start_byte=1,
            stop_byte=None,
            expected=-1,
            finite=False,
        )

    def test_base_magic_basic(self):
        tf = Field(
            "format",
            "name",
            start_byte=1,
            expected=4,
            fmt=">B",
        )

        with self.subTest(method="repr empty"):
            self.assertEqual(
                "<Field(EMPTY, fmt='>B')>", repr(tf)
            )

        tf.set(b"\x00\x01\x23\xae")
        with self.subTest(method="bytes"):
            self.assertEqual(b"\x00\x01\x23\xae", bytes(tf))
        with self.subTest(method="len"):
            self.assertEqual(4, len(tf))
        with self.subTest(method="str"):
            self.assertEqual("0 1 23 AE", str(tf))
        with self.subTest(method="repr"):
            self.assertEqual("<Field(0 1 23 AE, fmt='>B')>", repr(tf))

    def test_base_magic_additional(self):
        tf = Field(
            "format",
            "name",
            start_byte=0,
            expected=-1,
            fmt=">H"
        )
        tf.set(range(20))
        with self.subTest(method="repr"):
            self.assertEqual(
                "<Field(0 1 2 3 4 5 6 7 ...(12), fmt='>H')>", repr(tf)
            )

    def test_init_exc_float(self):
        with self.assertRaises(FieldContentError) as exc:
            Field(
                "format",
                "name",
                start_byte=0,
                expected=2,
                fmt=">H",
            ).set(b"\x01\x02\x03")
        self.assertEqual(
            "invalid content in Field: not integer count of words "
            "(expected 2, got 1.5)",
            exc.exception.args[0]
        )

    def test_init_exc_partial(self):
        with self.assertRaises(FieldContentError) as exc:
            Field(
                "format",
                "name",
                start_byte=0,
                expected=3,
                fmt=">H",
            ).set(b"\x01\x02")
        self.assertEqual(
            "invalid content in Field: fill ratio - 0.3",
            exc.exception.args[0]
        )

    def test_init_exc_partial_more(self):
        with self.assertRaises(FieldContentError) as exc:
            Field(
                "format",
                "name",
                start_byte=0,
                expected=3,
                fmt=">H",
            ).set(b"\x01\x02" * 5)
        self.assertEqual(
            "invalid content in Field: fill ratio - 1.7",
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
        tf = Field(
            "format",
            "name",
            start_byte=0,
            expected=-1,
            fmt=">f"
        )
        tf.set(1.263)
        self.assertEqual(
            b"\x3f\xa1\xa9\xfc",
            tf.content,
        )

    def test_set_empty(self):
        self.tf.set(b"")
        self.assertEqual(b"", self.tf.content)

    def test_set_not_supported(self):
        with self.assertRaises(TypeError) as exc:
            self.tf.set(type)
        self.assertEqual(
            "cannot convert 'type' object to bytes",
            exc.exception.args[0]
        )

    def test_extract_from(self):
        tf = Field(
            "format",
            "name",
            start_byte=2,
            expected=-1,
            fmt=">B"
        )
        tf.extract(b"\x01\x02\x03\x04\x05\x06")
        self.assertEqual(b"\x03\x04\x05\x06", tf.content)

        tf = Field(
            "format",
            "name",
            start_byte=2,
            expected=2,
            fmt=">B"
        )
        tf.extract(b"\x01\x02\x03\x04\x05\x06")
        self.assertEqual(b"\x03\x04", tf.content)

    def test_extract_from_empty(self):
        with self.assertRaises(ValueError) as exc:
            self.tf.extract(b"")
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

    def test_magic_str_float(self):
        tf = Field(
            "f", "n", start_byte=1, expected=-1, fmt=">e"
        )
        tf.set([12.2, -23])
        self.assertEqual("4A1A CDC0", str(tf))


class TestFieldSingle(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = SingleField("f", "n", start_byte=0, fmt=">H")

    def test_base_init(self):
        validate_object(
            self,
            SingleField(
                "format",
                "name",
                start_byte=1,
                fmt=">H",
            ),
            slice=slice(1, 3),
            mf_name="format",
            name="name",
            start_byte=1,
            stop_byte=3,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">H",
            bytesize=2,
            content=b"",
            default=b"",
            words_count=0,
            check_attrs=True,
            wo_attrs=["parent"],
        )

    def test_unpack(self):
        self.assertEqual(0, len(self.tf.unpack()))
        self.tf.set(0x123)
        self.assertListEqual([0x123], list(self.tf.unpack()))

    def test_madic_iter(self):
        self.tf.set(0xfa12)
        self.assertListEqual([0xfa12], [i for i in self.tf])


class TestFieldStatic(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = StaticField(
            "f", "n", start_byte=0, fmt=">H", default=0x1234
        )

    def test_base_init(self):
        validate_object(
            self,
            StaticField(
                "format",
                "name",
                start_byte=0,
                fmt=">I",
                default=0xfa1c
            ),
            slice=slice(0, 4),
            mf_name="format",
            name="name",
            start_byte=0,
            stop_byte=4,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">I",
            bytesize=4,
            content=b"\x00\x00\xfa\x1c",
            default=b"\x00\x00\xfa\x1c",
            words_count=1,
            check_attrs=True,
            wo_attrs=["parent"],
        )

    def test_set(self):
        with self.assertRaises(ValueError) as exc:
            self.tf.set(0x4321)
        self.assertEqual(
            "The current content of the static field is different from "
            r"the new content: b'\x124' != b'C!'",
            exc.exception.args[0]
        )

    def test_set_same(self):
        self.tf.set(0x1234)
        self.assertEqual(self.tf.content, b"\x12\x34")

    def test_set_empty(self):
        self.tf.set(b"")
        self.assertEqual(self.tf.content, b"\x12\x34")


class TestFieldAddress(unittest.TestCase):

    def test_base_init(self):
        validate_object(
            self,
            AddressField(
                "format",
                "address",
                start_byte=0,
                fmt=">I",
            ),
            slice=slice(0, 4),
            mf_name="format",
            name="address",
            start_byte=0,
            stop_byte=4,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">I",
            bytesize=4,
            content=b"",
            default=b"",
            words_count=0,
            check_attrs=True,
            wo_attrs=["parent"],
        )


class TestCrcField(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = CrcField("test", "crc", start_byte=0, fmt=">H")

    def test_base_init(self):
        field = CrcField(
                "format",
                "crc",
                start_byte=0,
                fmt=">H",
            )
        validate_object(
            self,
            field,
            slice=slice(0, 2),
            mf_name="format",
            name="crc",
            start_byte=0,
            stop_byte=2,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">H",
            bytesize=2,
            content=b"",
            default=b"",
            words_count=0,
            wo_fields=set(),
            algorithm=field.algorithm,
            algorithm_name="crc16-CCITT/XMODEM",
            check_attrs=True,
            wo_attrs=["parent"],
        )

    def test_calculate(self) -> None:
        msg = FieldMessage().configure(
            preamble=FieldSetter.static(fmt=">H", default=0xaa55),
            address=FieldSetter.address(fmt=">B"),
            data=FieldSetter.data(expected=-1, fmt=">B"),
            crc=FieldSetter.crc(fmt=">H", wo_fields={"preamble"}),
        ).set(
            address=1,
            data=[0] * 5
        )
        self.assertEqual(0x45a0, msg["crc"][0])

    def test_algorithms(self):
        check_data = {
            "crc16-CCITT/XMODEM": [
                (b"\x10\x01\x20\x04", 0x6af5),
                (bytes(range(15)), 0x9b92),
                (bytes(i % 256 for i in range(1500)), 0x9243),
                (b"\x01\x00\x00\x00\x00\x00", 0x45a0),
            ]
        }
        for name, algorithm in CrcField.CRC_ALGORITHMS.items():
            for content, ref in check_data[name]:
                with self.subTest(name=name, ref=ref):
                    self.assertEqual(ref, algorithm(content))


class TestFieldData(unittest.TestCase):

    def test_base_init(self):
        validate_object(
            self,
            DataField(
                "format",
                "data",
                start_byte=0,
                expected=2,
                fmt=">I",
            ),
            slice=slice(0, 8),
            mf_name="format",
            name="data",
            start_byte=0,
            stop_byte=8,
            expected=2,
            finite=True,
            may_be_empty=True,
            fmt=">I",
            bytesize=4,
            content=b"",
            default=b"",
            words_count=0,
            check_attrs=True,
            wo_attrs=["parent"],
        )


class TestFieldDataLength(unittest.TestCase):

    @staticmethod
    def get_tf(units: int, additive: int) -> DataLengthField:
        return DataLengthField(
            "f",
            "data_length",
            start_byte=0,
            fmt=">H",
            units=units,
            additive=additive
        )

    @staticmethod
    def get_tf_data(content):
        tf = DataField(
            "f",
            "data",
            start_byte=0,
            expected=2,
            fmt=">H",
        )
        tf.set(content)
        return tf

    def test_base_init(self):
        validate_object(
            self,
            DataLengthField(
                "format",
                "data_length",
                start_byte=0,
                fmt=">H",
                additive=0,
            ),
            slice=slice(0, 2),
            mf_name="format",
            name="data_length",
            start_byte=0,
            stop_byte=2,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">H",
            bytesize=2,
            content=b"",
            default=b"",
            words_count=0,
            behaviour="actual",
            units=Code.BYTES,
            additive=0,
            check_attrs=True,
            wo_attrs=["parent"],
        )

    def test_base_init_other(self):
        validate_object(
            self,
            DataLengthField(
                "format",
                "data_length",
                start_byte=0,
                fmt=">H",
                units=0x200,
                additive=10,
            ),
            slice=slice(0, 2),
            mf_name="format",
            name="data_length",
            start_byte=0,
            stop_byte=2,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">H",
            bytesize=2,
            content=b"",
            default=b"",
            words_count=0,
            behaviour="actual",
            units=Code.WORDS,
            additive=10,
            check_attrs=True,
            wo_attrs=["parent"],
        )

    def test_init_wrong_oper_core(self):
        with self.assertRaises(ValueError) as exc:
            DataLengthField(
                "f", "data_length", start_byte=0, fmt="b", units=Code.OK
            )
        self.assertEqual("invalid units: 1", exc.exception.args[0])

    def test_init_wrong_additive(self):
        with self.assertRaises(ValueError) as exc:
            DataLengthField(
                "f", "data_length", start_byte=0, fmt="b", additive=-1
            )
        self.assertEqual(
            "additive number must be positive integer, got -1",
            exc.exception.args[0]
        )

    def test_init_wrong_behaviour(self) -> None:
        with self.assertRaises(ValueError) as exc:
            DataLengthField(
                "f", "len", start_byte=0, fmt="b", behaviour="test"
            )
        self.assertEqual(
            "invalid behaviour: 'test' not in {'actual', 'expected2read}",
            exc.exception.args[0]
        )

    def test_update(self):
        tf_data = self.get_tf_data([0x12, 0x14])
        init_args = (
            (Code.BYTES, 0),
            (Code.BYTES, 4),
            (Code.WORDS, 0),
            (Code.WORDS, 7)
        )
        results = (4, 8, 2, 9)
        for i_test, (args, result) in enumerate(zip(init_args, results)):
            with self.subTest(i_test=i_test):
                tf = self.get_tf(*args)
                tf.set(tf.calculate(tf_data))
                self.assertEqual(
                    result,
                    tf.unpack()
                )

    def test_calculate_invalid_units(self):
        tf = self.get_tf(Code.BYTES, 0)
        tf._units = 0
        with self.assertRaises(ValueError) as exc:
            tf.calculate(self.get_tf_data([0, 1]))
        self.assertEqual("invalid units: 0", exc.exception.args[0])


class TestFieldOperation(unittest.TestCase):

    @staticmethod
    def get_tf(desc_dict=None) -> OperationField:
        return OperationField(
            "f", "operation", start_byte=0, fmt=">H", desc_dict=desc_dict
        )

    def test_base_init(self):
        tf = OperationField(
                "format",
                "operation",
                start_byte=0,
                fmt=">H"
            )
        tf.set(1)
        validate_object(
            self,
            tf,
            slice=slice(0, 2),
            mf_name="format",
            name="operation",
            start_byte=0,
            stop_byte=2,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">H",
            bytesize=2,
            content=b"\x00\x01",
            default=b"",
            words_count=1,
            base="w",
            desc="w",
            desc_dict={'r': 0, 'w': 1, 'e': 2},
            desc_dict_r={0: 'r', 1: 'w', 2: 'e'},
            check_attrs=True,
            wo_attrs=["parent"],
        )

    def test_base_init_custom_desc_dict(self):
        tf = OperationField(
                "format",
                "operation",
                start_byte=0,
                fmt=">H",
                desc_dict={"r1": 0x2, "r2": 0xf1, "w1": 0xf}
            )
        tf.set("w1")
        validate_object(
            self,
            tf,
            content=b"\x00\x0f",
            words_count=1,
            base="w",
            desc="w1",
            desc_dict={"r1": 0x2, "r2": 0xf1, "w1": 0x0f},
            desc_dict_r={0x2: "r1", 0xf1: "r2", 0x0f: "w1"},
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


class TestResponseField(unittest.TestCase):

    def setUp(self) -> None:
        self.tf = ResponseField(
            "test",
            "response",
            start_byte=2,
            fmt=">H",
            codes={
                0: Code.WAIT,
                1: ResponseField.OK,
                2: ResponseField.RAISE,
                3: ResponseField.WAIT,
            }
        )

    def test_init(self) -> None:
        self.tf.set(2)
        validate_object(
            self,
            self.tf,
            slice=slice(2, 4),
            mf_name="test",
            name="response",
            start_byte=2,
            stop_byte=4,
            expected=1,
            finite=True,
            may_be_empty=False,
            fmt=">H",
            bytesize=2,
            content=b"\x00\x02",
            default=b"\x00\x00",
            codes={
                0: Code.WAIT,
                1: ResponseField.OK,
                2: ResponseField.RAISE,
                3: ResponseField.WAIT,
            },
            default_code=Code.UNDEFINED,
            current_code=Code.RAISE,
            words_count=1,
            check_attrs=True,
            wo_attrs=["parent"],
        )
        self.assertEqual(
            self.tf.codes,
            {
                0: Code.WAIT,
                1: ResponseField.OK,
                2: ResponseField.RAISE,
                3: ResponseField.WAIT,
            }
        )
        self.assertEqual(Code.RAISE, self.tf.current_code)

    def test_current_code(self) -> None:
        for i_code, code in enumerate(
                (Code.WAIT, Code.OK, Code.RAISE, Code.WAIT, Code.UNDEFINED)
        ):
            self.tf.set(i_code)
            with self.subTest(content=i_code, code=code):
                self.assertEqual(self.tf, code)

    def test_raises(self) -> None:
        self.tf.set(b"")
        with self.subTest(exception="empty content"):
            with self.assertRaises(FieldContentError) as exc:
                self.tf.current_code
            self.assertEqual(
                "invalid content in ResponseField: content is empty",
                exc.exception.args[0]
            )

        with self.subTest(exception="undefined code"):
            tf = ResponseField(
                "test",
                "response",
                start_byte=2,
                fmt=">H",
                codes={
                    0: Code.WAIT,
                    1: ResponseField.OK,
                    2: ResponseField.RAISE,
                    3: ResponseField.WAIT,
                },
                default_code=None,
            )
            tf.set(10)
            with self.assertRaises(FieldContentError) as exc:
                tf.current_code
            self.assertEqual(
                "invalid content in ResponseField: "
                "undefined code by content 10",
                exc.exception.args[0]
            )

    def test_magic_eq(self) -> None:
        self.tf.set(2)
        self.assertTrue(self.tf == Code.RAISE)
        self.assertFalse(self.tf == 1)

    def test_magic_neq(self) -> None:
        self.tf.set(1)
        self.assertTrue(self.tf != Code.RAISE)
        self.assertFalse(self.tf != Code.OK)


class TestFieldSetter(unittest.TestCase):  # todo: test init field by FieldSetter

    def validate_setter(
            self,
            fs: FieldSetter,
            field_type: str = None,
            **kwargs: Any
    ):
        self.assertDictEqual(kwargs, fs.kwargs)
        self.assertEqual(field_type, fs.field_type)

    def test_init(self):
        self.validate_setter(FieldSetter(a=0, b=3), a=0, b=3)

    def test_init_with_default_bytes(self) -> None:
        with self.assertRaises(TypeError) as exc:
            FieldSetter(default=b"")
        self.assertEqual(
            "<class 'bytes'> not recommended bytes type for 'default' "
            "argument",
            exc.exception.args[0]
        )

    def test_base(self):
        self.validate_setter(
            FieldSetter.base(expected=1, fmt="i"),
            expected=1,
            fmt="i",
            default=[],
            may_be_empty=False,
        )

    def test_single(self):
        self.validate_setter(
            FieldSetter.single(fmt="i"),
            field_type="single",
            fmt="i",
            default=[],
            may_be_empty=False,
        )

    def test_static(self):
        self.validate_setter(
            FieldSetter.static(fmt="i", default=[]),
            field_type="static",
            fmt="i",
            default=[],
        )

    def test_address(self):
        self.validate_setter(
            FieldSetter.address(fmt="i"),
            field_type="address",
            fmt="i",
        )

    def test_crc(self) -> None:
        self.validate_setter(
            FieldSetter.crc(fmt="i"),
            field_type="crc",
            fmt="i",
            algorithm_name="crc16-CCITT/XMODEM",
            wo_fields=None
        )

    def test_data(self):
        self.validate_setter(
            FieldSetter.data(expected=3, fmt="i"),
            field_type="data",
            expected=3,
            fmt="i",
        )

    def test_data_length(self):
        self.validate_setter(
            FieldSetter.data_length(fmt="i"),
            field_type="data_length",
            fmt="i",
            behaviour="actual",
            additive=0,
            units=Code.BYTES,
        )

    def test_operation(self):
        self.validate_setter(
            FieldSetter.operation(fmt="i"),
            field_type="operation",
            fmt="i",
            desc_dict=None,
        )

    def test_response(self) -> None:
        self.validate_setter(
            FieldSetter.response(fmt="i", codes={0: Code.OK}),
            field_type="response",
            fmt="i",
            default=0,
            codes={0: Code.OK},
            default_code=Code.UNDEFINED,
        )

    def test_get_field_class(self) -> None:
        for field_type, ref in zip(
            [None] + list(FieldSetter.FIELD_TYPES),
            [
                Field,
                SingleField,
                StaticField,
                AddressField,
                CrcField,
                DataField,
                DataLengthField,
                OperationField,
                ResponseField,
            ]
        ):
            with self.subTest(field_type=field_type):
                res = FieldSetter(field_type=field_type).get_field_class()
                self.assertIs(res, ref)
                self.assertIn(
                    res,
                    get_args(FieldType),
                    f"FieldType not supports {res.__name__}"
                )
            break

    def test_init_kwargs(self) -> None:
        self.assertDictEqual(
            dict(field_type=None, a=3, b=3),
            FieldSetter(a=3, b=3).init_kwargs
        )

    def test_magic_eq(self) -> None:
        ref = FieldSetter(field_type="lol", a=5)
        res = FieldSetter(field_type="lol", a=5)
        self.assertEqual(ref, res)

        res2 = FieldSetter(a=5)
        self.assertNotEqual(ref, res2)
        self.assertNotEqual(ref, 5)

    def test_magic_repr_str(self) -> None:
        for ref, res in [
            ("<FieldSetter(field_type=None)>", FieldSetter()),
            ("<FieldSetter(field_type=a, a=19)>", FieldSetter(field_type="a", a=19))
        ]:
            self.assertEqual(ref, repr(res))
            self.assertEqual(ref, str(res))

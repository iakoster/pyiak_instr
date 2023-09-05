import unittest

from numpy.testing import assert_array_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import ContentError
from src.pyiak_instr.testing import validate_object

from tests.pyiak_instr_ti.store import TIField, TIStruct


class TestField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            name="std",
            bytes_expected=0,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=True,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, None),
            start=0,
            stop=None,
            fmt_bytesize=1,
            words_expected=0,
            wo_attrs=["codec"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="empty 'name'"):
            with self.assertRaises(ValueError) as exc:
                self._instance(name="")
            self.assertEqual("empty 'name' not allowed", exc.exception.args[0])

        with self.subTest(test="'bytes_expected' is a negative"):
            with self.assertRaises(ValueError) as exc:
                self._instance(bytes_expected=-1)
            self.assertEqual(
                "'bytes_expected' can't be a negative number",
                exc.exception.args[0],
            )

        with self.subTest(test="'stop' is equal to zero"):
            with self.assertRaises(ValueError) as exc:
                self._instance(stop=0)
            self.assertEqual(
                "'stop' can't be equal to zero", exc.exception.args[0]
            )

        with self.subTest(test="'stop' and 'bytes_expected' setting"):
            with self.assertRaises(TypeError) as exc:
                self._instance(stop=1, bytes_expected=1)
            self.assertEqual(
                "'bytes_expected' and 'stop' setting not allowed",
                exc.exception.args[0],
            )

        with self.subTest(
                test="'bytes_expected' is not comparable with 'word_bytesize'"
        ):
            with self.assertRaises(ValueError) as exc:
                self._instance(bytes_expected=5, fmt=Code.U16)
            self.assertEqual(
                "'bytes_expected' does not match an integer word count",
                exc.exception.args[0],
            )

        with self.subTest(test="'bytes_expected' more than negative start"):
            with self.assertRaises(ValueError) as exc:
                self._instance(start=-2, bytes_expected=3)
            self.assertEqual(
                "it will be out of bounds",
                exc.exception.args[0],
            )

        with self.subTest(test="default changes"):
            self._instance(stop=5, default=b"aaaaa")
            self._instance(stop=4, fmt=Code.U16, default=b"aaaa")
            with self.assertRaises(ValueError):
                self._instance(fmt=Code.U16, default=b"aaa")
            with self.assertRaises(ValueError) as exc:
                self._instance(stop=4, default=b"aaa")
            self.assertEqual(
                "default value is incorrect",
                exc.exception.args[0],
            )

    def test_init_start_stop(self) -> None:
        cases = (
            ((0, None, 0), dict()),
            ((0, 2, 2), dict(bytes_expected=2)),
            ((2, None, 0), dict(start=2)),
            ((2, 5, 3), dict(start=2, stop=5)),
            ((2, -2, 0), dict(start=2, stop=-2)),
            ((-4, -2, 2), dict(start=-4, bytes_expected=2)),
            ((-4, -2, 2), dict(start=-4, stop=-2)),
            ((-2, None, 2), dict(start=-2)),
            ((-2, None, 2), dict(start=-2, bytes_expected=2)),
            ((0, 2, 2), dict(stop=2)),
            ((-6, -3, 3), dict(start=-6, stop=-3)),
            ((0, -3, 0), dict(stop=-3)),
            ((0, 2, 2), dict(bytes_expected=2)),
            ((-6, -3, 3), dict(start=-6, bytes_expected=3)),
            ((-2, None, 2), dict(start=-2, bytes_expected=2)),
        )
        for case, ((start, stop, expected), kw) in enumerate(cases):
            with self.subTest(case=case):
                fld = TIField(**kw)
                self.assertEqual(start, fld.start)
                self.assertEqual(stop, fld.stop)
                self.assertEqual(expected, fld.bytes_expected)

    def test_decode(self) -> None:
        assert_array_equal(
            [0, 2], self._instance().decode(b"\x00\x02")
        )

    def test_decode_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            self._instance(fmt=Code.U16).decode(b"\x00", verify=True)
        self.assertEqual(
            "invalid content in TIField: <Code.INVALID_LENGTH: 1026>",
            exc.exception.args[0],
        )

    def test_encode(self) -> None:
        assert_array_equal(
            b"\x00\x02", self._instance().encode([0, 2])
        )

    def test_encode_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            self._instance(fmt=Code.U16).encode(b"\x01", verify=True)
        self.assertEqual(
            "invalid content in TIField: <Code.INVALID_LENGTH: 1026>",
            exc.exception.args[0],
        )

    def test_extract(self) -> None:
        field = self._instance(start=2, stop=3)
        self.assertEqual(b"\x02", field.extract(bytes(range(10))))
        self.assertEqual(b"", field.extract(b""))

    def test_verify(self) -> None:
        obj = self._instance(fmt=Code.U16)
        self.assertIs(Code.OK, obj.verify(b"\x01\x02"))
        self.assertIs(Code.INVALID_LENGTH, obj.verify(b"\x01\x02\x03"))

        self.assertIs(Code.OK, obj.verify(b"\x01\x02", raise_if_false=True))
        with self.assertRaises(ContentError) as exc:
            obj.verify(b"\x01\x02\x03", raise_if_false=True)
        self.assertEqual(
            "invalid content in TIField: <Code.INVALID_LENGTH: 1026>",
            exc.exception.args[0]
        )

        obj = self._instance(stop=4, fmt=Code.U16)
        self.assertIs(Code.OK, obj.verify(b"\xff" * 4))
        self.assertIs(Code.INVALID_LENGTH, obj.verify(b"\x01\x02\x03"))

        obj = self._instance(start=-1)
        self.assertEqual(1, obj.bytes_expected)
        with self.assertRaises(ContentError) as exc:
            obj.verify(b"ff", raise_if_false=True)
        self.assertEqual(
            "invalid content in TIField: <Code.INVALID_LENGTH: 1026>",
            exc.exception.args[0],
        )

    def test_default(self) -> None:
        self.assertEqual(
            b"a" * 10, self._instance(stop=10, default=b"a").default
        )

    @staticmethod
    def _instance(
            name: str = "std",
            start: int = 0,
            stop: int | None = None,
            bytes_expected: int = 0,
            fmt: Code = Code.U8,
            default: bytes = b"",
    ) -> TIField:
        return TIField(
            name=name,
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            default=default,
        )


class TestStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            name="std",
            dynamic_field_name="f2",
            minimum_size=7,
            is_dynamic=True,
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="without fields"):
            with self.assertRaises(ValueError) as exc:
                TIStruct()
            self.assertEqual(
                "TIStruct without fields not allowed", exc.exception.args[0]
            )

        with self.subTest(test="empty field name"):
            with self.assertRaises(KeyError) as exc:
                TIStruct(fields={"": TIField()})
            self.assertEqual(
                "empty field name not allowed", exc.exception.args[0]
            )

        with self.subTest(test="wrong field name"):
            with self.assertRaises(KeyError) as exc:
                TIStruct(fields={"f0": TIField()})
            self.assertEqual(
                "invalid struct name: 'f0' != 'std'", exc.exception.args[0]
            )

        with self.subTest(test="two dynamic"):
            with self.assertRaises(TypeError) as exc:
                TIStruct(
                    fields={
                        "f0": TIField(name="f0"),
                        "f1": TIField(name="f1"),
                    }
                )
            self.assertEqual(
                "two dynamic fields not allowed", exc.exception.args[0]
            )

    def test_change(self) -> None:
        instance = self._instance()
        content = bytearray(range(20))
        instance.change(content, "f2", b"a")
        self.assertEqual(b"\x00\x01\x02a\x10\x11\x12\x13", content)

    def test_decode(self) -> None:
        with self.subTest(test="decode one field"):
            assert_array_equal(
                [0, 2, 4, 6],
                self._instance().decode("f2", b"\x00\x02\x04\x06"),
            )

        with self.subTest(test="decode all"):
            ref = {
                "f0": [0],
                "f1": [1, 2],
                "f2": [3, 4, 5],
                "f3": [6, 7, 8],
                "f4": [9],
            }
            res = self._instance().decode(bytes(range(10)))

            for field in ref:
                with self.subTest(field=field):
                    assert_array_equal(ref[field], res[field])

    def test_decode_exc(self) -> None:
        with self.subTest(test="with kwargs"):
            with self.assertRaises(TypeError) as exc:
                self._instance().decode(content=b"")
            self.assertEqual(
                "takes no keyword arguments", exc.exception.args[0]
            )

        with self.subTest(test="invalid argument"):
            with self.assertRaises(TypeError) as exc:
                self._instance().decode(b"", b"")
            self.assertEqual("invalid arguments", exc.exception.args[0])

    def test_encode(self) -> None:
        with self.subTest(test="bytes"):
            self.assertEqual(
                {
                    "f0": b"\x00",
                    "f1": b"\x01\x02",
                    "f2": b"\x03\x04\x05",
                    "f3": b"\x06\x07\x08",
                    "f4": b"\x09",
                },
                self._instance().encode(bytes(range(10))),
            )

        with self.subTest(test="fields"):
            self.assertEqual(
                {"f0": b"\x00", "f2": b"\x03\x04\x05", "f3": b"\x06\x07\x08"},
                self._instance().encode(f0=0, f2=[3, 4, 5], f3=[6, 7, 8]),
            )

        with self.subTest(test="with only auto fields"):
            self.assertEqual(
                dict(
                    f0=b"aa",
                    f1=b"",
                    f2=b"dd",
                ),
                self._instance(
                    f0=TIField(name="f0", stop=2, default=b"aa"),
                    f1=TIField(name="f1", start=2, stop=-2),
                    f2=TIField(name="f2", start=-2, default=b"d"),
                ).encode(all_fields=True)
            )

    def test_encode_all_fields(self) -> None:
        obj = self._instance()
        with self.subTest(test="full"):
            self.assertDictEqual(
                dict(
                    f0=b"\x00",
                    f1=b"\x01\x02",
                    f2=bytes(range(3, 6)),
                    f3=b"\x06\x07\x08",
                    f4=b"\x09",
                ),
                obj.encode(
                    all_fields=True,
                    f0=0,
                    f1=[1, 2],
                    f2=bytes(range(3, 6)),
                    f3=[6, 7, 8],
                    f4=9,
                )
            )

        with self.subTest(test="without default"):
            self.assertDictEqual(
                dict(
                    f0=b"\xfa",
                    f1=b"\x01\x02",
                    f2=bytes(range(3, 6)),
                    f3=b"\x06\x07\x08",
                    f4=b"\x09",
                ),
                obj.encode(
                    all_fields=True,
                    f1=[1, 2],
                    f2=bytes(range(3, 6)),
                    f3=[6, 7, 8],
                    f4=9,
                )
            )

        with self.subTest(test="without infinite"):
            self.assertDictEqual(
                dict(
                    f0=b"\xfa",
                    f1=b"\x01\x02",
                    f2=b"",
                    f3=b"\x06\x07\x08",
                    f4=b"\x09",
                ),
                obj.encode(
                    all_fields=True,
                    f1=[1, 2],
                    f3=[6, 7, 8],
                    f4=9,
                )
            )

        with self.subTest(test="without fill"):
            self.assertDictEqual(
                dict(
                    f0=b"\xfa",
                    f1=b"\xff\xff",
                    f2=b"",
                    f3=b"\x06\x07\x08",
                    f4=b"\x09",
                ),
                obj.encode(
                    all_fields=True,
                    f3=[6, 7, 8],
                    f4=9
                )
            )

    def test_encode_exc(self) -> None:
        with self.subTest(test="with args and kwargs"):
            with self.assertRaises(TypeError) as exc:
                self._instance().encode(b"", f0=[])
            self.assertEqual(
                "takes a bytes or fields (both given)", exc.exception.args[0]
            )

        with self.subTest(test="short content"):
            with self.assertRaises(ContentError) as exc:
                self._instance().encode(b"aaaa")
            self.assertEqual(
                "invalid content in TIStruct: <Code.INVALID_LENGTH: 1026> "
                "- expected at least 7, got 4",
                exc.exception.args[0],
            )

        with self.subTest(test="long content"):
            with self.assertRaises(ContentError) as exc:
                self._instance(
                    f0=TIField(name="f0", stop=2)
                ).encode(b"aaaa")
            self.assertEqual(
                "invalid content in TIStruct: <Code.INVALID_LENGTH: 1026> "
                "- expected 2, got 4",
                exc.exception.args[0],
            )

        with self.subTest(test="two args"):
            with self.assertRaises(TypeError) as exc:
                self._instance().encode(1, 2)
            self.assertEqual("invalid arguments count (got 2)", exc.exception.args[0])

    def test_extract(self) -> None:
        instance = self._instance()
        content = bytes(range(10))

        with self.subTest(test="extract all"):
            self.assertDictEqual(
                {
                    "f0": b"\x00",
                    "f1": b"\x01\x02",
                    "f2": b"\x03\x04\x05",
                    "f3": b"\x06\x07\x08",
                    "f4": b"\t",
                },
                instance.extract(content)
            )

        with self.subTest(test="extract one"):
            self.assertEqual(b"\x01\x02", instance.extract(content, "f1"))

        with self.subTest(test="extract several"):
            self.assertDictEqual(
                {
                    "f1": b"\x01\x02",
                    "f2": b"\x03\x04\x05",
                },
                instance.extract(content, "f1", "f2")
            )

    def test_verify(self) -> None:
        with self.subTest(test="correct content"):
            self._instance().verify(bytes(range(10)), verify_fields=True)

    def test_items(self) -> None:
        obj = self._instance()
        for ref, (res, parser) in zip(obj._f, obj.items()):
            self.assertEqual(ref, res)
            self.assertIsInstance(parser, TIField)

    def test__verify_fields_list(self) -> None:
        obj = self._instance()
        with self.subTest(test="all takes"):
            obj._verify_fields_list({"f0", "f1", "f2", "f3", "f4"})

        with self.subTest(test="without one"):
            with self.assertRaises(AttributeError) as exc:
                obj._verify_fields_list({"f0", "f1", "f2", "f3"})
            self.assertEqual(
                "missing or extra fields were found: 'f4'",
                exc.exception.args[0]
            )

        with self.subTest(test="with extra"):
            with self.assertRaises(AttributeError) as exc:
                obj._verify_fields_list({"f0", "f1", "f2", "f3", "f4", "f5"})
            self.assertEqual(
                "missing or extra fields were found: 'f5'",
                exc.exception.args[0]
            )

        with self.subTest(test="without with default"):
            obj._verify_fields_list({"f0", "f1", "f3", "f4"})

        with self.subTest(test="without infinite"):
            obj._verify_fields_list({"f1", "f3", "f4"})

    def test_magic_contains(self) -> None:
        self.assertTrue("f0" in self._instance())
        self.assertFalse("six" in self._instance())

    def test_magic_getitem(self) -> None:
        ref = self._instance()["f2"]
        self.assertEqual("f2", ref.name)
        self.assertEqual(slice(3, -4), ref.slice_)

    def test_magic_iter(self) -> None:
        name = ""
        for name, struct in zip(
                ["f0", "f1", "f2", "f3", "f4"],
                self._instance(),
        ):
            with self.subTest(field=name):
                self.assertEqual(name, struct.name)
        self.assertEqual(name, "f4")

    @staticmethod
    def _instance(
            **fields: TIField
    ) -> TIStruct:
        if len(fields) == 0:
            fields = dict(
                f0=TIField(
                    name="f0",
                    start=0,
                    default=b"\xfa",
                    stop=1,
                ),
                f1=TIField(
                    name="f1", start=1, bytes_expected=2, default=b"\xff"
                ),
                f2=TIField(
                    name="f2", start=3, stop=-4
                ),
                f3=TIField(
                    name="f3", start=-4, stop=-1
                ),
                f4=TIField(
                    name="f4", start=-1, stop=None
                ),
            )
        return TIStruct(fields=fields)

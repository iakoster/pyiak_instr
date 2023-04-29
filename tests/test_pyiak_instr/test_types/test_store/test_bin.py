from __future__ import annotations

import shutil
import unittest
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, TypeAlias

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.types import PatternABC
from src.pyiak_instr.types.store import (
    STRUCT_DATACLASS,
    BytesFieldABC,
    BytesFieldPatternABC,
    BytesFieldStructProtocol,
    BytesStorageABC,
    BytesStoragePatternABC,
    ContinuousBytesStoragePatternABC,
)

from ...env import TEST_DATA_DIR
from ....utils import validate_object, compare_objects


DATA_DIR = TEST_DATA_DIR / __name__.split(".")[-1]


@STRUCT_DATACLASS
class TIStruct(BytesFieldStructProtocol):

    def decode(self, content: bytes) -> npt.NDArray[np.int_ | np.float_]:
        return np.frombuffer(
            content, np.uint8 if self.fmt is Code.U8 else np.uint16
        )

    def encode(self, content: int | float | Iterable[int | float]) -> bytes:
        return np.array(content).astype(
            np.uint8 if self.fmt is Code.U8 else np.uint16
        ).tobytes()

    def _verify_values_before_modifying(self) -> None:
        if self.fmt not in {Code.U8, Code.U16}:
            raise ValueError("invalid fmt")
        if self.order is not Code.BIG_ENDIAN:
            raise ValueError("invalid order")
        super()._verify_values_before_modifying()

    @property
    def word_bytesize(self) -> int:
        if self.fmt is Code.U16:
            return 2
        return 1


class TIField(BytesFieldABC["TIStorage", TIStruct]):
    ...


class TIStorage(BytesStorageABC[TIField, TIStruct]):
    _struct_field = {TIStruct: TIField}


class TIPattern(BytesFieldPatternABC[TIStruct]):

    _options = {"basic": TIStruct}


class TIStoragePattern(BytesStoragePatternABC[TIStorage, TIPattern]):

    _options = {"basic": TIStorage}
    _sub_p_type = TIPattern


class TIContinuousStoragePattern(
    ContinuousBytesStoragePatternABC[TIStorage, TIPattern]
):
    _options = {"basic": TIStorage}
    _sub_p_type = TIPattern


class TestBytesFieldStructProtocol(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
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
        )
        for case, ((start, stop, expected), kw) in enumerate(cases):
            with self.subTest(case=case):
                res = TIStruct(**kw)
                self.assertEqual(start, res.start)
                self.assertEqual(stop, res.stop)
                self.assertEqual(expected, res.bytes_expected)

    def test_verify(self) -> None:
        obj = self._instance(fmt=Code.U16)
        self.assertTrue(obj.verify(b"\x01\x02"))
        self.assertFalse(obj.verify(b"\x01\x02\x03"))

        obj = self._instance(stop=4, fmt=Code.U16)
        self.assertTrue(obj.verify(b"\xff" * 4))
        self.assertFalse(obj.verify(b"\x01\x02\x03"))

        obj = self._instance(start=-1)
        self.assertEqual(1, obj.bytes_expected)
        self.assertFalse(obj.verify(b"ff"))

    def test_magic_post_init(self) -> None:

        with self.subTest(test="'bytes_expected' < 0"):
            self.assertEqual(
                0, self._instance(bytes_expected=-255).bytes_expected
            )

        cases = [
            (dict(stop=2), 2),
            (dict(start=-6, stop=-3), 3),
            (dict(stop=-3), 0),  # dynamic - slice(0, -3)
        ]
        for i_case, (kwargs, bytes_expected) in enumerate(cases):
            with self.subTest(test="stop is not None", case=i_case):
                self.assertEqual(
                    bytes_expected,
                    self._instance(**kwargs).bytes_expected,
                )

        cases = [
            (dict(bytes_expected=2), 2),
            (dict(start=-6, bytes_expected=3), -3),
            (dict(start=-2, bytes_expected=2), None),
        ]
        for i_case, (kwargs, stop) in enumerate(cases):
            with self.subTest(test="'bytes_expected' > 0", case=i_case):
                self.assertEqual(stop, self._instance(**kwargs).stop)

    def test_magic_post_init_exc(self) -> None:
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
                print(self._instance(start=-2, bytes_expected=3))
            self.assertEqual(
                "it will be out of bounds",
                exc.exception.args[0],
            )

    @staticmethod
    def _instance(
            start: int = 0,
            stop: int | None = None,
            bytes_expected: int = 0,
            fmt: Code = Code.U8,
    ) -> BytesFieldStructProtocol:
        return TIStruct(
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
        )


class TestBytesFieldABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            bytes_count=10,
            content=bytes(range(10)),
            is_empty=False,
            name="test",
            struct=TIStruct(),
            words_count=10,
        )

    def test_decode(self) -> None:
        assert_array_equal([*range(10)], self._instance().decode())

    def test_encode(self) -> None:
        obj = self._instance()
        obj.encode(0xff)
        self.assertEqual(b"\xff", obj.content)

    def test_verify(self) -> None:
        self.assertTrue(self._instance(stop=5).verify(b"\x00\x02\x05\x55\xaa"))
        self.assertFalse(self._instance(stop=5).verify(b"\x00"))

    def test_magic_bytes(self) -> None:
        self.assertEqual(bytes(range(10)), bytes(self._instance()))

    def test_magic_getitem(self) -> None:
        self.assertEqual(2, self._instance()[2])
        assert_array_equal([0, 1, 2], self._instance()[:3])

    def test_magic_iter(self) -> None:
        for ref, res in zip(range(10), self._instance()):
            self.assertEqual(ref, res)

    def test_magic_len(self) -> None:
        self.assertEqual(10, len(self._instance()))

    def _instance(self, stop: int | None = None) -> TIField:
        fields = {"test": TIStruct(stop=stop)}
        if stop is not None:
            fields["ph"] = TIStruct(start=stop, stop=None)
        storage = TIStorage("std", fields).encode(bytes(range(10)))
        return storage["test"]


class TestBytesStorageABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance,
            content=b"",
            is_dynamic=True,
            minimum_size=7,
            name="test_storage",
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIStorage("test", {})
        self.assertEqual(
            "TIStorage without fields are forbidden", exc.exception.args[0]
        )

    def test_change(self) -> None:
        obj = self._instance.encode(
                second=[1, 2],
                fourth=(3, 4, 5),
                fifth=6,
            )
        self.assertEqual(b"\xfa" + bytes(range(1, 7)), obj.content)

        obj.change("first", b"\x00")
        self.assertEqual(bytes(range(7)), obj.content)

    def test_change_exc(self) -> None:
        obj = self._instance
        with self.subTest(test="change in empty message"):
            with self.assertRaises(TypeError) as exc:
                obj.change("", b"")
            self.assertEqual("message is empty", exc.exception.args[0])

    def test_decode(self) -> None:
        obj = self._instance
        for name, decoded in obj.decode().items():
            assert_array_equal([], decoded, err_msg=f"{name} not empty")

        content = dict(
            first=[0],
            second=[1, 2],
            third=[3, 4, 5],
            fourth=[6, 7, 8],
            fifth=[9],
        )
        obj.encode(bytes(range(10)))
        for name, decoded in obj.decode().items():
            with self.subTest(field=name):
                assert_array_equal(content[name], decoded)

    def test_encode_set_all(self) -> None:
        with self.subTest(test="full"):
            obj = self._instance.encode(
                first=b"\x00",
                second=b"\x01\x02",
                third=bytes(range(3, 6)),
                fourth=b"\x06\x07\x08",
                fifth=b"\x09",
            )
            self.assertEqual(bytes(range(10)), obj.content)

            for (_, ref), (name, res) in zip(
                    dict(
                        first=b"\x00",
                        second=b"\x01\x02",
                        third=bytes(range(3, 6)),
                        fourth=b"\x06\x07\x08",
                        fifth=b"\x09",
                    ).items(), obj.items()
            ):
                with self.subTest(name=name):
                    self.assertEqual(ref, res.content)

        with self.subTest(test="without default"):
            self.assertEqual(
                b"\xfa" + bytes(range(1, 10)),
                self._instance.encode(
                    second=b"\x01\x02",
                    third=bytes(range(3, 6)),
                    fourth=b"\x06\x07\x08",
                    fifth=b"\x09",
                ).content
            )

        with self.subTest(test="without infinite"):
            self.assertEqual(
                bytes(range(7)),
                self._instance.encode(
                    first=0,
                    second=[1, 2],
                    fourth=(3, 4, 5),
                    fifth=6.0,
                ).content
            )

    def test_encode_change(self) -> None:
        obj = self._instance.encode(
            second=[1, 2],
            fourth=(3, 4, 5),
            fifth=6,
        )
        self.assertEqual(b"\xfa\x01\x02\x03\x04\x05\x06", obj.content)
        obj.encode(first=b"\xaa", fourth=[5, 4, 3])
        self.assertEqual(b"\xaa\x01\x02\x05\x04\x03\x06", obj.content)
        obj["third"].encode([0xaa, 0, 0x77])
        self.assertEqual(
            b"\xaa\x01\x02\xaa\x00\x77\x05\x04\x03\x06", obj.content
        )

    def test_encode_extract(self) -> None:
        with self.subTest(test="extract all"):
            obj = self._instance.encode(bytes(range(20)))
            for (name, ref), res in zip(
                    dict(
                        first=b"\x00",
                        second=b"\x01\x02",
                        third=bytes(range(3, 16)),
                        fourth=b"\x10\x11\x12",
                        fifth=b"\x13",
                    ).items(), obj
            ):
                with self.subTest(name=name):
                    self.assertEqual(ref, res.content)

        with self.subTest(test="without_infinite"):
            obj = self._instance.encode(bytes(range(7)))
            self.assertEqual(bytes(range(7)), obj.content)
            for (name, ref), res in zip(
                    dict(
                        first=b"\x00",
                        second=b"\x01\x02",
                        third=b"",
                        fourth=b"\x03\x04\x05",
                        fifth=b"\x06",
                    ).items(), obj
            ):
                with self.subTest(name=name):
                    self.assertEqual(ref, res.content)

        with self.subTest(test="drop past message"):
            obj = self._instance.encode(bytes(range(7)))
            self.assertEqual(bytes(range(7)), obj.content)

            obj.encode(bytes(range(7, 0, -1)))
            self.assertEqual(bytes(range(7, 0, -1)), obj.content)

    def test_encode_exc(self) -> None:
        with self.subTest(test="encode with message and fields"):
            with self.assertRaises(TypeError) as exc:
                self._instance.encode(b"\x01", first=21)
            self.assertEqual(
                "takes a message or fields (both given)",
                exc.exception.args[0],
            )

        with self.subTest(test="encode empty message"):
            with self.assertRaises(TypeError) as exc:
                self._instance.encode(b"")
            self.assertEqual("message is empty", exc.exception.args[0])

        with self.subTest(test="encode short message"):
            with self.assertRaises(ValueError) as exc:
                self._instance.encode(b"\xff")
            self.assertEqual("bytes content too short", exc.exception.args[0])

        with self.subTest(test="encode long message"):
            with self.assertRaises(ValueError) as exc:
                TIStorage(
                    "std", {"test": TIStruct(stop=2)}
                ).encode(b"\xff" * 4)
            self.assertEqual("bytes content too long", exc.exception.args[0])

    def test_items(self) -> None:
        obj = self._instance
        for ref, (res, parser) in zip(obj._f, obj.items()):
            self.assertEqual(ref, res)
            self.assertIsInstance(parser, TIField)

    def test__check_fields_list(self) -> None:
        obj = self._instance
        with self.subTest(test="all takes"):
            obj._check_fields_list({
                "first", "second", "third", "fourth", "fifth"
            })

        with self.subTest(test="without one"):
            with self.assertRaises(AttributeError) as exc:
                obj._check_fields_list({
                    "first", "third", "fourth", "fifth"
                })
            self.assertEqual(
                "missing or extra fields were found: 'second'",
                exc.exception.args[0]
            )

        with self.subTest(test="with extra"):
            with self.assertRaises(AttributeError) as exc:
                obj._check_fields_list({
                    "first", "second", "third", "fourth", "fifth", "sixth"
                })
            self.assertEqual(
                "missing or extra fields were found: 'sixth'",
                exc.exception.args[0]
            )

        with self.subTest(test="without with default"):
            obj._check_fields_list({
                "second", "third", "fourth", "fifth"
            })

        with self.subTest(test="without infinite"):
            obj._check_fields_list({
                "second", "fourth", "fifth"
            })

    def test__encode_content(self) -> None:
        parser = self._instance["second"]
        func = self._instance._encode_content

        self.assertEqual(b"ab", func(parser, b"ab"))
        self.assertEqual(b"\x00\xff", func(parser, [0, 255]))

    def test__encode_content_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            self._instance._encode_content(self._instance["second"], 0)
        self.assertEqual(
            "'00' is not correct for 'second'", exc.exception.args[0]
        )

    def test_magic_contains(self) -> None:
        self.assertTrue("first" in self._instance)
        self.assertFalse("six" in self._instance)

    def test_magic_getitem(self) -> None:
        ref = self._instance["third"]
        self.assertEqual("third", ref.name)
        self.assertEqual(slice(3, -4), ref.struct.slice_)

    def test_magic_iter(self) -> None:
        name = ""
        for name, parser in zip([
            "first", "second", "third", "fourth", "fifth",
        ], self._instance):
            with self.subTest(field=name):
                self.assertEqual(name, parser.name)
        self.assertEqual(name, "fifth")

    def test_magic_len(self) -> None:
        obj = self._instance
        self.assertEqual(0, len(obj))
        obj.encode(b"f" * 255)
        self.assertEqual(255, len(obj))

    @property
    def _instance(self) -> TIStorage:
        return TIStorage("test_storage", dict(
            first=TIStruct(start=0, default=b"\xfa", stop=1),
            second=TIStruct(start=1, bytes_expected=2),
            third=TIStruct(start=3, stop=-4),
            fourth=TIStruct(start=-4, stop=-1),
            fifth=TIStruct(start=-1, stop=None),
        ))


class TestBytesStoragePatternABC(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_write(self) -> None:
        path = DATA_DIR / "test_write.ini"
        self._instance.write(path)

        ref = [
            "[test]",
            r"test = \dct(typename,basic,name,test,val,\tpl(11))",
            r"first = \dct(typename,basic,bytes_expected,0,int,3,"
            r"list,\lst(2,3,4))",
            r"second = \dct(typename,basic,bytes_expected,0,boolean,True)",
            r"third = \dct(typename,basic,bytes_expected,0,"
            r"dict,\dct(0,1,2,3))",
        ]
        i_line = 0
        with open(path, "r") as file:
            for ref, res in zip(ref, file.read().split("\n")):
                i_line += 1
                with self.subTest(test="new", line=i_line):
                    self.assertEqual(ref, res)
        self.assertEqual(5, i_line)

        TIStoragePattern(typename="basic", name="test", val=(11,)).configure(
            first=TIPattern(typename="basic", bytes_expected=0, int=11),
        ).write(path)

        ref = [
            "[test]",
            r"test = \dct(typename,basic,name,test,val,\tpl(11))",
            r"first = \dct(typename,basic,bytes_expected,0,int,11)",
        ]
        i_line = 0
        with open(path, "r") as file:
            for ref, res in zip(ref, file.read().split("\n")):
                i_line += 1
                with self.subTest(test="rewrite", line=i_line):
                    self.assertEqual(ref, res)
        self.assertEqual(3, i_line)

    def test_write_exc_not_configured(self) -> None:
        with self.assertRaises(NotConfiguredYet) as exc:
            TIStoragePattern(typename="basic", name="test", val=(11,)).write(
                DATA_DIR / "test.ini"
            )
        self.assertEqual(
            "TIStoragePattern not configured yet", exc.exception.args[0]
        )

    def test_write_read(self) -> None:
        path = DATA_DIR / "test_write_read.ini"
        ref = self._instance
        ref.write(path)
        res = TIStoragePattern.read(path, "test")

        self.assertIsNot(ref, res)
        self.assertEqual(ref, res)
        self.assertEqual(ref._sub_p, res._sub_p)

    def test_read_many_keys(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIStoragePattern.read(DATA_DIR, "key_1", "key_2")
        self.assertEqual("given 2 keys, expect one", exc.exception.args[0])

    @property
    def _instance(self) -> TIStoragePattern:
        return TIStoragePattern(
            typename="basic", name="test", val=(11,)
        ).configure(
            first=TIPattern(
                typename="basic", bytes_expected=0, int=3, list=[2, 3, 4]
            ),
            second=TIPattern(
                typename="basic", bytes_expected=0, boolean=True
            ),
            third=TIPattern(
                typename="basic", bytes_expected=0, dict={0: 1, 2: 3}
            )
        )


class TestContinuousBytesStoragePatternABC(unittest.TestCase):

    def test_get(self) -> None:
        ref = TIStorage(name="test", fields=dict(
                f0=TIStruct(start=0, bytes_expected=4),
                f1=TIStruct(start=4, bytes_expected=2),
                f2=TIStruct(start=6, stop=-7),
                f3=TIStruct(start=-7, bytes_expected=3),
                f4=TIStruct(start=-4, bytes_expected=4),
        ))
        res = TIContinuousStoragePattern(
                typename="basic", name="test"
        ).configure(
            f0=TIPattern(typename="basic", bytes_expected=4),
            f1=TIPattern(typename="basic", bytes_expected=2),
            f2=TIPattern(typename="basic", bytes_expected=0),
            f3=TIPattern(typename="basic", bytes_expected=3),
            f4=TIPattern(typename="basic", bytes_expected=4),
        ).get()
        compare_objects(self, ref, res)
        for ref_field, res_field in zip(ref, res):
            with self.subTest(field=ref_field.name):
                compare_objects(self, ref_field.struct, res_field.struct)

    def test__modify_all(self) -> None:
        with self.subTest(test="dyn in middle"):
            self.assertDictEqual(
                dict(
                    f0=dict(start=0),
                    f1=dict(start=4),
                    f2=dict(start=6, stop=-7),
                    f3=dict(start=-7),
                    f4=dict(start=-4),
                ),
                TIContinuousStoragePattern(
                    typename="basic", name="test"
                ).configure(
                    f0=TIPattern(typename="basic", bytes_expected=4),
                    f1=TIPattern(typename="basic", bytes_expected=2),
                    f2=TIPattern(typename="basic", bytes_expected=0),
                    f3=TIPattern(typename="basic", bytes_expected=3),
                    f4=TIPattern(typename="basic", bytes_expected=4),
                )._modify_all(True, {})
            )

        with self.subTest(test="last dyn"):
            self.assertDictEqual(
                dict(
                    f0=dict(start=0),
                    f1=dict(start=4),
                    f2=dict(start=6, stop=None, req=1),
                ),
                TIContinuousStoragePattern(
                    typename="basic", name="test"
                ).configure(
                    f0=TIPattern(typename="basic", bytes_expected=4),
                    f1=TIPattern(typename="basic", bytes_expected=2),
                    f2=TIPattern(typename="basic", bytes_expected=0),
                )._modify_all(True, {"f2": {"req": 1}})
            )

        with self.subTest(test="only dyn"):
            self.assertDictEqual(
                dict(
                    f0=dict(start=0, stop=None),
                ),
                TIContinuousStoragePattern(
                    typename="basic", name="test"
                ).configure(
                    f0=TIPattern(typename="basic", bytes_expected=0),
                )._modify_all(True, {})
            )

        with self.subTest(test="without dyn"):
            self.assertDictEqual(
                dict(
                    f0=dict(start=0),
                    f1=dict(start=4),
                    f2=dict(start=6),
                ),
                TIContinuousStoragePattern(
                    typename="basic", name="test"
                ).configure(
                    f0=TIPattern(typename="basic", bytes_expected=4),
                    f1=TIPattern(typename="basic", bytes_expected=2),
                    f2=TIPattern(typename="basic", bytes_expected=3),
                )._modify_all(True, {})
            )

    def test__modify_all_exc(self) -> None:
        with self.subTest(test="two dynamic field"):
            with self.assertRaises(TypeError) as exc:
                TIContinuousStoragePattern(
                    typename="basic", name="test",
                ).configure(
                    f0=TIPattern(typename="basic", bytes_expected=0),
                    f1=TIPattern(typename="basic", bytes_expected=0),
                )._modify_all(True, {})
            self.assertEqual(
                "two dynamic field not allowed", exc.exception.args[0]
            )

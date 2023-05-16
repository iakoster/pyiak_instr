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
from src.pyiak_instr.types import Encoder
from src.pyiak_instr.types.store import (
    STRUCT_DATACLASS,
    BytesFieldStructABC,
    BytesStorageABC,
    BytesStorageStructABC,
)

from tests.test_pyiak_instr.env import TEST_DATA_DIR
from tests.utils import validate_object, compare_objects
from .ti import TIEncoder, TIFieldStruct, TIStorageStruct, TIStorage


DATA_DIR = TEST_DATA_DIR / __name__.split(".")[-1]


# class TIPattern(BytesFieldPatternABC[TIStruct]):
#
#     _options = {"basic": TIStruct}
#
#
# class TIStoragePattern(BytesStoragePatternABC[TIStorage, TIPattern]):
#
#     _options = {"basic": TIStorage}
#     _sub_p_type = TIPattern
#
#
# class TIContinuousStoragePattern(
#     ContinuousBytesStoragePatternABC[TIStorage, TIPattern]
# ):
#     _options = {"basic": TIStorage}
#     _sub_p_type = TIPattern
#
#
# class TestBytesFieldABC(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             self._instance(),
#             bytes_count=10,
#             content=bytes(range(10)),
#             is_empty=False,
#             name="test",
#             struct=TIStruct(),
#             words_count=10,
#         )
#
#     def test_decode(self) -> None:
#         assert_array_equal([*range(10)], self._instance().decode())
#
#     def test_encode(self) -> None:
#         obj = self._instance()
#         obj.encode(0xff)
#         self.assertEqual(b"\xff", obj.content)
#
#     def test_verify(self) -> None:
#         self.assertTrue(self._instance(stop=5).verify(b"\x00\x02\x05\x55\xaa"))
#         self.assertFalse(self._instance(stop=5).verify(b"\x00"))
#
#     def test_magic_bytes(self) -> None:
#         self.assertEqual(bytes(range(10)), bytes(self._instance()))
#
#     def test_magic_getitem(self) -> None:
#         self.assertEqual(2, self._instance()[2])
#         assert_array_equal([0, 1, 2], self._instance()[:3])
#
#     def test_magic_iter(self) -> None:
#         for ref, res in zip(range(10), self._instance()):
#             self.assertEqual(ref, res)
#
#     def test_magic_len(self) -> None:
#         self.assertEqual(10, len(self._instance()))
#
#     def test_magic_str(self) -> None:
#         cases = (
#             ("empty", "TIField(EMPTY)", {"content": b""}),
#             ("not empty", "TIField(1 203 405 607 809)", {"fmt": Code.U16}),
#             (
#                 "large u8",
#                 "TIField(0 1 2 3 ... 1C 1D 1E 1F)",
#                 {"content": bytes(range(32))},
#             ),
#             (
#                 "large u16",
#                 "TIField(1 203 405 ... 1A1B 1C1D 1E1F)",
#                 {"content": bytes(range(32)), "fmt": Code.U16},
#             ),
#             (
#                 "large u24",
#                 "TIField(102 30405 ... 1B1C1D 1E1F20)",
#                 {"content": bytes(range(33)), "fmt": Code.U24},
#             ),
#             (
#                 "large u40",
#                 "TIField(1020304 ... 1E1F202122)",
#                 {"content": bytes(range(35)), "fmt": Code.U40},
#             ),
#         )
#
#         for test, ref, kw in cases:
#             with self.subTest(test=test):
#                 self.assertEqual(ref, str(self._instance(**kw)))
#
#     @staticmethod
#     def _instance(
#             stop: int | None = None,
#             fmt: Code = Code.U8,
#             content: bytes = bytes(range(10)),
#     ) -> TIField:
#         fields = {"test": TIStruct(stop=stop, fmt=fmt)}
#         if stop is not None:
#             fields["ph"] = TIStruct(start=stop, stop=None)
#
#         storage = TIStorage("std", fields)
#         if len(content):
#             storage.encode(content)
#         return storage["test"]


class TestBytesStorageABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            minimum_size=7,
            wo_attrs=["struct"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIStorage({})
        self.assertEqual("TIStorage without fields", exc.exception.args[0])

    # def test_change(self) -> None:
    #     obj = self._instance.encode(
    #             second=[1, 2],
    #             fourth=(3, 4, 5),
    #             fifth=6,
    #         )
    #     self.assertEqual(b"\xfa" + bytes(range(1, 7)), obj.content)
    #
    #     obj.change("first", b"\x00")
    #     self.assertEqual(bytes(range(7)), obj.content)
    #
    # def test_change_exc(self) -> None:
    #     obj = self._instance
    #     with self.subTest(test="change in empty message"):
    #         with self.assertRaises(TypeError) as exc:
    #             obj.change("", b"")
    #         self.assertEqual("message is empty", exc.exception.args[0])
    #
    # def test_encode_change(self) -> None:
    #     obj = self._instance.encode(
    #         second=[1, 2],
    #         fourth=(3, 4, 5),
    #         fifth=6,
    #     )
    #     self.assertEqual(b"\xfa\x01\x02\x03\x04\x05\x06", obj.content)
    #     obj.encode(first=b"\xaa", fourth=[5, 4, 3])
    #     self.assertEqual(b"\xaa\x01\x02\x05\x04\x03\x06", obj.content)
    #     obj["third"].encode([0xaa, 0, 0x77])
    #     self.assertEqual(
    #         b"\xaa\x01\x02\xaa\x00\x77\x05\x04\x03\x06", obj.content
    #     )
    #
    # def test_encode_extract(self) -> None:
    #     with self.subTest(test="extract all"):
    #         obj = self._instance.encode(bytes(range(20)))
    #         for (name, ref), res in zip(
    #                 dict(
    #                     first=b"\x00",
    #                     second=b"\x01\x02",
    #                     third=bytes(range(3, 16)),
    #                     fourth=b"\x10\x11\x12",
    #                     fifth=b"\x13",
    #                 ).items(), obj
    #         ):
    #             with self.subTest(name=name):
    #                 self.assertEqual(ref, res.content)
    #
    #     with self.subTest(test="without_infinite"):
    #         obj = self._instance.encode(bytes(range(7)))
    #         self.assertEqual(bytes(range(7)), obj.content)
    #         for (name, ref), res in zip(
    #                 dict(
    #                     first=b"\x00",
    #                     second=b"\x01\x02",
    #                     third=b"",
    #                     fourth=b"\x03\x04\x05",
    #                     fifth=b"\x06",
    #                 ).items(), obj
    #         ):
    #             with self.subTest(name=name):
    #                 self.assertEqual(ref, res.content)
    #
    #     with self.subTest(test="drop past message"):
    #         obj = self._instance.encode(bytes(range(7)))
    #         self.assertEqual(bytes(range(7)), obj.content)
    #
    #         obj.encode(bytes(range(7, 0, -1)))
    #         self.assertEqual(bytes(range(7, 0, -1)), obj.content)
    #
    def test_magic_len(self) -> None:
        obj = self._instance()
        self.assertEqual(0, len(obj))
        # obj.encode(b"f" * 255)
        # self.assertEqual(255, len(obj))
    #
    # def test_magic_str(self) -> None:
    #     with self.subTest(test="empty"):
    #         self.assertEqual(
    #             "TIStorage(EMPTY)", str(self._instance)
    #         )
    #
    #     with self.subTest(test="not empty"):
    #         self.assertEqual(
    #             "TIStorage(first=0, second=1 2, "
    #             "third=3 4 5 6 ... C0 C1 C2 C3, fourth=C4 C5 C6, fifth=C7)",
    #             str(self._instance.encode(bytes(range(200)))),
    #         )

    @staticmethod
    def _instance() -> TIStorage:
        return TIStorage(dict(
            f0=TIFieldStruct(
                name="f0", start=0, default=b"\xfa", stop=1, encoder=TIEncoder
            ),
            f1=TIFieldStruct(
                name="f1", start=1, bytes_expected=2, encoder=TIEncoder
            ),
            f2=TIFieldStruct(name="f2", start=3, stop=-4, encoder=TIEncoder),
            f3=TIFieldStruct(name="f3", start=-4, stop=-1, encoder=TIEncoder),
            f4=TIFieldStruct(
                name="f4", start=-1, stop=None, encoder=TIEncoder
            ),
        ), name="test")


# class TestBytesStoragePatternABC(unittest.TestCase):
#
#     @classmethod
#     def tearDownClass(cls) -> None:
#         if TEST_DATA_DIR.exists():
#             shutil.rmtree(TEST_DATA_DIR)
#
#     def test_init(self) -> None:
#         obj = self._instance
#         self.assertIs(obj, obj["pattern"])
#
#     def test_write(self) -> None:
#         path = DATA_DIR / "test_write.ini"
#         self._instance.write(path)
#
#         ref = [
#             "[test]",
#             r"test = \dct(typename,basic,name,test,val,\tpl(11))",
#             r"first = \dct(typename,basic,bytes_expected,0,int,3,"
#             r"list,\lst(2,3,4))",
#             r"second = \dct(typename,basic,bytes_expected,0,boolean,True)",
#             r"third = \dct(typename,basic,bytes_expected,0,"
#             r"dict,\dct(0,1,2,3))",
#         ]
#         i_line = 0
#         with open(path, "r") as file:
#             for ref, res in zip(ref, file.read().split("\n")):
#                 i_line += 1
#                 with self.subTest(test="new", line=i_line):
#                     self.assertEqual(ref, res)
#         self.assertEqual(5, i_line)
#
#         TIStoragePattern(typename="basic", name="test", val=(11,)).configure(
#             first=TIPattern(typename="basic", bytes_expected=0, int=11),
#         ).write(path)
#
#         ref = [
#             "[test]",
#             r"test = \dct(typename,basic,name,test,val,\tpl(11))",
#             r"first = \dct(typename,basic,bytes_expected,0,int,11)",
#         ]
#         i_line = 0
#         with open(path, "r") as file:
#             for ref, res in zip(ref, file.read().split("\n")):
#                 i_line += 1
#                 with self.subTest(test="rewrite", line=i_line):
#                     self.assertEqual(ref, res)
#         self.assertEqual(3, i_line)
#
#     def test_write_exc_not_configured(self) -> None:
#         with self.assertRaises(NotConfiguredYet) as exc:
#             TIStoragePattern(typename="basic", name="test", val=(11,)).write(
#                 DATA_DIR / "test.ini"
#             )
#         self.assertEqual(
#             "TIStoragePattern not configured yet", exc.exception.args[0]
#         )
#
#     def test_write_read(self) -> None:
#         path = DATA_DIR / "test_write_read.ini"
#         ref = self._instance
#         ref.write(path)
#         res = TIStoragePattern.read(path, "test")
#
#         self.assertIsNot(ref, res)
#         self.assertEqual(ref, res)
#         self.assertEqual(ref._sub_p, res._sub_p)
#
#     def test_read_many_keys(self) -> None:
#         with self.assertRaises(TypeError) as exc:
#             TIStoragePattern.read(DATA_DIR, "key_1", "key_2")
#         self.assertEqual("given 2 keys, expect one", exc.exception.args[0])
#
#     @property
#     def _instance(self) -> TIStoragePattern:
#         return TIStoragePattern(
#             typename="basic", name="test", val=(11,)
#         ).configure(
#             first=TIPattern(
#                 typename="basic", bytes_expected=0, int=3, list=[2, 3, 4]
#             ),
#             second=TIPattern(
#                 typename="basic", bytes_expected=0, boolean=True
#             ),
#             third=TIPattern(
#                 typename="basic", bytes_expected=0, dict={0: 1, 2: 3}
#             )
#         )
#
#
# class TestContinuousBytesStoragePatternABC(unittest.TestCase):
#
#     def test_get(self) -> None:
#         pattern = TIContinuousStoragePattern(
#             typename="basic", name="test"
#         ).configure(
#             f0=TIPattern(typename="basic", bytes_expected=4),
#             f1=TIPattern(typename="basic", bytes_expected=2),
#             f2=TIPattern(typename="basic", bytes_expected=0),
#             f3=TIPattern(typename="basic", bytes_expected=3),
#             f4=TIPattern(typename="basic", bytes_expected=4),
#         )
#
#         ref = TIStorage(name="test", fields=dict(
#                 f0=TIStruct(start=0, bytes_expected=4),
#                 f1=TIStruct(start=4, bytes_expected=2),
#                 f2=TIStruct(start=6, stop=-7),
#                 f3=TIStruct(start=-7, bytes_expected=3),
#                 f4=TIStruct(start=-4, bytes_expected=4),
#         ), pattern=pattern)
#         res = pattern.get()
#         compare_objects(self, ref, res)
#         for ref_field, res_field in zip(ref, res):
#             with self.subTest(field=ref_field.name):
#                 compare_objects(self, ref_field.struct, res_field.struct)
#
#     def test__modify_all(self) -> None:
#         with self.subTest(test="dyn in middle"):
#             self.assertDictEqual(
#                 dict(
#                     f0=dict(start=0),
#                     f1=dict(start=4),
#                     f2=dict(start=6, stop=-7),
#                     f3=dict(start=-7),
#                     f4=dict(start=-4),
#                 ),
#                 TIContinuousStoragePattern(
#                     typename="basic", name="test"
#                 ).configure(
#                     f0=TIPattern(typename="basic", bytes_expected=4),
#                     f1=TIPattern(typename="basic", bytes_expected=2),
#                     f2=TIPattern(typename="basic", bytes_expected=0),
#                     f3=TIPattern(typename="basic", bytes_expected=3),
#                     f4=TIPattern(typename="basic", bytes_expected=4),
#                 )._modify_all(True, {})
#             )
#
#         with self.subTest(test="last dyn"):
#             self.assertDictEqual(
#                 dict(
#                     f0=dict(start=0),
#                     f1=dict(start=4),
#                     f2=dict(start=6, stop=None, req=1),
#                 ),
#                 TIContinuousStoragePattern(
#                     typename="basic", name="test"
#                 ).configure(
#                     f0=TIPattern(typename="basic", bytes_expected=4),
#                     f1=TIPattern(typename="basic", bytes_expected=2),
#                     f2=TIPattern(typename="basic", bytes_expected=0),
#                 )._modify_all(True, {"f2": {"req": 1}})
#             )
#
#         with self.subTest(test="only dyn"):
#             self.assertDictEqual(
#                 dict(
#                     f0=dict(start=0, stop=None),
#                 ),
#                 TIContinuousStoragePattern(
#                     typename="basic", name="test"
#                 ).configure(
#                     f0=TIPattern(typename="basic", bytes_expected=0),
#                 )._modify_all(True, {})
#             )
#
#         with self.subTest(test="without dyn"):
#             self.assertDictEqual(
#                 dict(
#                     f0=dict(start=0),
#                     f1=dict(start=4),
#                     f2=dict(start=6),
#                 ),
#                 TIContinuousStoragePattern(
#                     typename="basic", name="test"
#                 ).configure(
#                     f0=TIPattern(typename="basic", bytes_expected=4),
#                     f1=TIPattern(typename="basic", bytes_expected=2),
#                     f2=TIPattern(typename="basic", bytes_expected=3),
#                 )._modify_all(True, {})
#             )
#
#     def test__modify_all_exc(self) -> None:
#         with self.subTest(test="two dynamic field"):
#             with self.assertRaises(TypeError) as exc:
#                 TIContinuousStoragePattern(
#                     typename="basic", name="test",
#                 ).configure(
#                     f0=TIPattern(typename="basic", bytes_expected=0),
#                     f1=TIPattern(typename="basic", bytes_expected=0),
#                 )._modify_all(True, {})
#             self.assertEqual(
#                 "two dynamic field not allowed", exc.exception.args[0]
#             )

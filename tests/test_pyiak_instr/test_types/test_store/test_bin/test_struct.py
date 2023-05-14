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
from src.pyiak_instr.types.store._bin._struct import (
    STRUCT_DATACLASS,
    BytesFieldStructABC,
    BytesStorageStructABC,
)

from tests.test_pyiak_instr.env import TEST_DATA_DIR
from tests.utils import validate_object, compare_objects


class TIEncoder(Encoder):

    def __init__(self, fmt: Code = Code.U8, order: Code = Code.BIG_ENDIAN):
        if fmt not in {
            Code.U8, Code.U16, Code.U24, Code.U32, Code.U40
        }:
            raise ValueError("invalid fmt")
        if order is not Code.BIG_ENDIAN:
            raise ValueError("invalid order")
        self.fmt, self.order = fmt, order

    def decode(self, value: bytes) -> npt.NDArray[np.int_ | np.float_]:
        return np.frombuffer(
            value, np.uint8 if self.fmt is Code.U8 else np.uint16
        )

    def encode(self, value: int | float | Iterable[int | float]) -> bytes:
        if isinstance(value, bytes):
            return value
        return np.array(value).astype(
            np.uint8 if self.fmt is Code.U8 else np.uint16
        ).tobytes()

    @property
    def value_size(self) -> int:
        return (
            Code.U8, Code.U16, Code.U24, Code.U32, Code.U40
        ).index(self.fmt) + 1


@STRUCT_DATACLASS
class TIFieldStruct(BytesFieldStructABC):
    ...


@STRUCT_DATACLASS
class TIStorageStruct(BytesStorageStructABC[TIFieldStruct]):
    ...


class TestBytesFieldStructABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            name="",
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
            wo_attrs=["encoder"],
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
                res = TIFieldStruct(**kw, encoder=TIEncoder)
                self.assertEqual(start, res.start)
                self.assertEqual(stop, res.stop)
                self.assertEqual(expected, res.bytes_expected)

    def test_decode(self) -> None:
        assert_array_equal(
            [0, 2], self._instance().decode(b"\x00\x02")
        )

    def test_encode(self) -> None:
        assert_array_equal(
            b"\x00\x02", self._instance().encode([0, 2])
        )

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
                self._instance(start=-2, bytes_expected=3)
            self.assertEqual(
                "it will be out of bounds",
                exc.exception.args[0],
            )

        with self.subTest(test="without encoder"):
            with self.assertRaises(ValueError) as exc:
                TIFieldStruct()
            self.assertEqual(
                "struct encoder not specified", exc.exception.args[0]
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

    @staticmethod
    def _instance(
            start: int = 0,
            stop: int | None = None,
            bytes_expected: int = 0,
            fmt: Code = Code.U8,
            default: bytes = b"",
    ) -> BytesFieldStructABC:
        return TIFieldStruct(
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            fmt=fmt,
            encoder=TIEncoder,
            default=default,
        )


class TestBytesStorageStructABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            name="std",
            fields={},
            dynamic_field_name="f2",
            minimum_size=7,
            is_dynamic=True,
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="without fields"):
            with self.assertRaises(ValueError) as exc:
                TIStorageStruct()
            self.assertEqual(
                "TIStorageStruct without fields", exc.exception.args[0]
            )

        with self.subTest(test="empty field name"):
            with self.assertRaises(KeyError) as exc:
                TIStorageStruct(fields={"": TIFieldStruct(encoder=TIEncoder)})
            self.assertEqual(
                "empty field name not allowed", exc.exception.args[0]
            )

        with self.subTest(test="wrong field name"):
            with self.assertRaises(KeyError) as exc:
                TIStorageStruct(fields={"f0": TIFieldStruct(encoder=TIEncoder)})
            self.assertEqual(
                "invalid struct name: 'f0' != ''", exc.exception.args[0]
            )

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

    def test_encode_exc(self) -> None:
        with self.subTest(test="with args and kwargs"):
            with self.assertRaises(TypeError) as exc:
                self._instance().encode(b"", f0=[])
            self.assertEqual(
                "takes a bytes or fields (both given)", exc.exception.args[0]
            )

        with self.subTest(test="without args and kwargs"):
            with self.assertRaises(TypeError) as exc:
                self._instance().encode()
            self.assertEqual("missing arguments", exc.exception.args[0])

        with self.subTest(test="empty content"):
            with self.assertRaises(ValueError) as exc:
                self._instance().encode(b"")
            self.assertEqual("content is empty", exc.exception.args[0])

    def test_items(self) -> None:
        obj = self._instance()
        for ref, (res, parser) in zip(obj._f, obj.items()):
            self.assertEqual(ref, res)
            self.assertIsInstance(parser, TIFieldStruct)

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
            **fields: TIFieldStruct
    ) -> TIStorageStruct:
        if len(fields) == 0:
            fields = dict(
                f0=TIFieldStruct(
                    name="f0",
                    start=0,
                    default=b"\xfa",
                    stop=1,
                    encoder=TIEncoder,
                ),
                f1=TIFieldStruct(
                    name="f1", start=1, bytes_expected=2, encoder=TIEncoder
                ),
                f2=TIFieldStruct(
                    name="f2", start=3, stop=-4, encoder=TIEncoder
                ),
                f3=TIFieldStruct(
                    name="f3", start=-4, stop=-1, encoder=TIEncoder
                ),
                f4=TIFieldStruct(
                    name="f4", start=-1, stop=None, encoder=TIEncoder
                ),
            )
        return TIStorageStruct(fields=fields)

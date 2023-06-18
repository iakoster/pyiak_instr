from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.encoders.types import Encoder

from ....env import TEST_DATA_DIR
from .....utils import validate_object, compare_objects

from tests.pyiak_instr_ti.store import TIField, TIStruct, TIContainer


DATA_DIR = TEST_DATA_DIR / __name__.split(".")[-1]


class TestBytesStorageABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            has_pattern=False,
            wo_attrs=["struct"],
        )

    def test_decode(self) -> None:
        obj = self._instance().encode(bytes(range(10)))
        for (name, ref), (rs_name, res) in zip(dict(
            f0=[0], f1=[1, 2], f2=[3, 4, 5], f3=[6, 7, 8], f4=[9]
        ).items(), obj.decode().items()):
            with self.subTest(test="decode all", field=name):
                assert_array_equal(ref, res)

        with self.subTest(test="f2"):
            assert_array_equal([3, 4, 5], obj.decode("f2"))

    def test_decode_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            self._instance().decode(1)
        self.assertEqual("invalid arguments", exc.exception.args[0])

    def test_encode(self) -> None:
        obj = self._instance().encode(bytes(range(10)))
        self._verify_content(
            obj,
            f0=b"\x00",
            f1=b"\x01\x02",
            f2=b"\x03\x04\x05",
            f3=b"\x06\x07\x08",
            f4=b"\x09",
        )

        obj.encode(f1=b"\xff\xfa", f2=b"abcsddd")
        self._verify_content(
            obj,
            f0=b"\x00",
            f1=b"\xff\xfa",
            f2=b"abcsddd",
            f3=b"\x06\x07\x08",
            f4=b"\x09",
        )

        obj.encode(bytes(range(10, 20)))
        self._verify_content(
            obj,
            f0=b"\x0a",
            f1=b"\x0b\x0c",
            f2=b"\x0d\x0e\x0f",
            f3=b"\x10\x11\x12",
            f4=b"\x13",
        )

    def test_bytes_count(self) -> None:
        obj = self._instance()
        with self.subTest(test="empty"):
            self.assertEqual(0, obj.bytes_count())
            self.assertEqual(0, obj.bytes_count("f0"))

        obj.encode(bytes(range(7)))
        with self.subTest(test="filled"):
            self.assertEqual(7, obj.bytes_count())
            self.assertEqual(2, obj.bytes_count("f1"))
            self.assertEqual(0, obj.bytes_count("f2"))

    def test_content(self) -> None:
        obj = self._instance()
        with self.subTest(test="empty"):
            self.assertEqual(b"", obj.content())
            self.assertEqual(b"", obj.content("f0"))

        obj.encode(bytes(range(7)))
        with self.subTest(test="filled"):
            self.assertEqual(bytes(range(7)), obj.content())
            self.assertEqual(b"\x01\x02", obj.content("f1"))
            self.assertEqual(b"", obj.content("f2"))

    def test_is_empty(self) -> None:
        obj = self._instance()
        with self.subTest(test="empty"):
            self.assertEqual(True, obj.is_empty())
            self.assertEqual(True, obj.is_empty("f0"))

        obj.encode(bytes(range(7)))
        with self.subTest(test="filled"):
            self.assertEqual(False, obj.is_empty())
            self.assertEqual(False, obj.is_empty("f1"))
            self.assertEqual(True, obj.is_empty("f2"))

    def test_words_count(self) -> None:
        obj = self._instance()
        with self.subTest(test="empty"):
            self.assertDictEqual(dict(
                f0=0, f1=0, f2=0, f3=0, f4=0
            ), obj.words_count())
            self.assertEqual(0, obj.words_count("f0"))

        obj.encode(bytes(range(7)))
        with self.subTest(test="filled"):
            self.assertEqual(dict(
                f0=1, f1=2, f2=0, f3=3, f4=1
            ), obj.words_count())
            self.assertEqual(2, obj.words_count("f1"))
            self.assertEqual(0, obj.words_count("f2"))

    def test_words_count_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            self._instance().words_count("", "")
        self.assertEqual("invalid arguments", exc.exception.args[0])

    def test_magic_bytes(self) -> None:
        obj = self._instance()
        self.assertEqual(b"", bytes(obj))
        obj.encode(b"f" * 255)
        self.assertEqual(b"f" * 255, bytes(obj))

    def test_magic_len(self) -> None:
        obj = self._instance()
        self.assertEqual(0, len(obj))
        obj.encode(b"f" * 255)
        self.assertEqual(255, len(obj))

    def test_magic_str(self) -> None:
        obj = TIContainer(TIStruct(
            fields=dict(
                f0=TIField(
                    name="f0",
                    start=0,
                    default=b"\xfa",
                    stop=1,
                ),
                f1=TIField(
                    name="f1",
                    start=1,
                    bytes_expected=2,
                    fmt=Code.U16,
                ),
                f2=TIField(
                    name="f2",
                    start=3,
                    stop=-4,
                    fmt=Code.U40,
                ),
                f3=TIField(
                    name="f3",
                    start=-4,
                    stop=-1,
                ),
                f4=TIField(
                    name="f4", start=-1, stop=None
                ),
            ),
            name="test",
        ))

        with self.subTest(test="empty"):
            self.assertEqual("TIContainer(EMPTY)", str(obj))

        with self.subTest(test="with one empty field"):
            self.assertEqual(
                "TIContainer(f0=0, f1=102, f2=EMPTY, f3=3 4 5, f4=6)",
                str(obj.encode(bytes(range(7)))),
            )

        with self.subTest(test="large"):
            self.assertEqual(
                "TIContainer(f0=0, f1=102, f2=304050607 ... BCBDBEBFC0, "
                "f3=C1 C2 C3, f4=C4)",
                str(obj.encode(bytes(range(197)))),
            )

        obj = TIContainer(TIStruct(
            fields={"f0": TIField(name="f0")}
        ))
        with self.subTest(test="infinite u8"):
            self.assertEqual(
                "TIContainer(f0=0 1 2 3 ... 11 12 13 14)",
                str(obj.encode(bytes(range(21)))),
            )

        obj = TIContainer(TIStruct(
            fields={"f0": TIField(name="f0", fmt=Code.U16)}
        ))
        with self.subTest(test="infinite u16"):
            self.assertEqual(
                "TIContainer(f0=1 203 405 ... 1011 1213 1415)",
                str(obj.encode(bytes(range(22)))),
            )

        obj = TIContainer(TIStruct(fields={"f0": TIField(name="f0", fmt=Code.U24)}))
        with self.subTest(test="infinite u24"):
            self.assertEqual(
                "TIContainer(f0=102 30405 ... 151617 18191A)",
                str(obj.encode(bytes(range(27)))),
            )

    def _verify_content(self, res: TIContainer, **fields: bytes) -> None:
        content = b""
        for field, ref in fields.items():
            with self.subTest(field=field):
                self.assertEqual(ref, res.content(field))
            content += ref
        with self.subTest(field="all"):
            self.assertEqual(content, res.content())

    @staticmethod
    def _instance() -> TIContainer:
        return TIContainer(TIStruct(
            fields=dict(
                f0=TIField(
                    name="f0",
                    start=0,
                    default=b"\xfa",
                    stop=1,
                ),
                f1=TIField(
                    name="f1", start=1, bytes_expected=2
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
            ),
            name="test",
        ))

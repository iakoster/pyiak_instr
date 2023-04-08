from __future__ import annotations

import shutil
import unittest
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal

from src.pyiak_instr.utilities import split_complex_dict
from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.types import PatternABC
from src.pyiak_instr.types.store import (
    BytesFieldABC,
    BytesFieldStructProtocol,
    BytesStorageABC,
    BytesStoragePatternABC,
    ContinuousBytesStoragePatternABC,
)

from ...env import TEST_DATA_DIR
from ....utils import validate_object


DATA_DIR = TEST_DATA_DIR / __name__.split(".")[-1]


@dataclass(frozen=True, kw_only=True)
class Struct(BytesFieldStructProtocol):

    start: int = 0

    stop: int | None = None

    bytes_expected: int = 0

    default: bytes = b""

    word_bytesize_: int = 1

    def decode(self, content: bytes) -> npt.NDArray[np.int_ | np.float_]:
        return np.frombuffer(content, np.uint8)

    def encode(self, content: int | float | Iterable[int | float]) -> bytes:
        return np.array(content).astype(np.uint8).tobytes()

    def validate(self, content: bytes) -> bool:
        length = len(content)
        if self.stop is None:
            return length == abs(self.start)
        if self.start >= 0 and self.stop > 0 or self.start < 0 and self.stop < 0:
            return length == (self.stop - self.start)
        if self.start >= 0 > self.stop:
            return True

    @property
    def word_bytesize(self) -> int:
        return self.word_bytesize_


class Field(BytesFieldABC[Struct]):

    content_: bytes = bytes([1, 2, 3, 4, 5])

    @property
    def content(self) -> bytes:
        return self.content_


class FieldFull(BytesFieldABC[Struct]):

    def __init__(
            self,
            storage: Storage | StorageKwargs,
            name: str,
            struct: Struct,
    ):
        super().__init__(name, struct)
        self._storage = storage

    @property
    def content(self) -> bytes:
        return self._storage.content[self.struct.slice_]


class Storage(
    BytesStorageABC[FieldFull, Struct]
):

    def __getitem__(self, name: str) -> FieldFull:
        return FieldFull(self, name, self._f[name])


class StorageKwargs(BytesStorageABC[FieldFull, Struct]):

    def __init__(
            self,
            name: str,
            fields: dict[str, Struct],
            **kwargs: Any,
    ):
        super().__init__(name=name, fields=fields)
        self.kwargs = kwargs

    def __getitem__(self, name: str) -> FieldFull:
        return FieldFull(self, name, self._f[name])


class FieldPattern(PatternABC[Struct]):

    _options = {"base": Struct}

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> Struct:
        return Struct(**self._kw, **additions)


class StoragePattern(BytesStoragePatternABC[Storage, FieldPattern]):

    _options = {"base": Storage}
    _sub_p_type = FieldPattern

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> Storage:
        raise NotImplementedError()


# class ContinuousStoragePattern(
#     ContinuousBytesStoragePatternABC[Storage, FieldPattern, Struct]
# ):
#
#     _options = {"base": Storage}
#     _sub_p_type = FieldPattern
#
#     def get(
#         self, changes_allowed: bool = False, **additions: Any
#     ) -> Storage:
#         raise NotImplementedError()


class TestBytesFieldStructProtocol(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            bytes_expected=0,
            default=b"",
            has_default=False,
            is_floating=True,
            slice_=slice(0, None),
            start=0,
            stop=None,
            word_bytesize=1,
            word_bytesize_=1,
            words_expected=0,
        )

    def test_magic_post_init(self) -> None:

        with self.subTest(test="'bytes_expected' < 0"):
            self.assertEqual(
                0, self._instance(bytes_expected=-255).bytes_expected
            )

        cases = [
            (dict(stop=2), 2),
            (dict(start=-6, stop=-3), 3),
            (dict(stop=-3), 0),
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
        with self.subTest(test="'stop' == 0"):
            with self.assertRaises(ValueError) as exc:
                self._instance(stop=0)
            self.assertEqual(
                "'stop' can't be equal to zero", exc.exception.args[0]
            )

        with self.subTest(test="'stop' and 'bytes_expected' setting"):
            with self.assertRaises(TypeError) as exc:
                self._instance(stop=1, bytes_expected=1)
            self.assertEqual(
                "'bytes_expected' or 'stop' setting allowed",
                exc.exception.args[0],
            )

        with self.subTest(
                test="'bytes_expected' is not comparable with 'word_bytesize'"
        ):
            with self.assertRaises(ValueError) as exc:
                self._instance(bytes_expected=5, word_bytesize=2)
            self.assertEqual(
                "'bytes_expected' does not match an integer word count",
                exc.exception.args[0],
            )

    @staticmethod
    def _instance(
            start: int = 0,
            stop: int | None = None,
            bytes_expected: int = 0,
            word_bytesize: int = 1
    ) -> BytesFieldStructProtocol:
        return Struct(
            start=start,
            stop=stop,
            bytes_expected=bytes_expected,
            word_bytesize_=word_bytesize,
        )


class TestBytesFieldABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance,
            bytes_count=5,
            content=b"\x01\x02\x03\x04\x05",
            content_=b"\x01\x02\x03\x04\x05",
            name="test",
            struct=Struct(stop=5),
            words_count=5,
        )

    def test_decode(self) -> None:
        assert_array_equal([1, 2, 3, 4, 5], self._instance.decode())

    def test_encode(self) -> None:
        self.assertEqual(b"\x01\x02", self._instance.encode([1, 2]))

    def test_validate(self) -> None:
        self.assertTrue(self._instance.validate(b"\x00\x02\x05\x55\xaa"))
        self.assertFalse(self._instance.validate(b"\x00"))

    def test_magic_bytes(self) -> None:
        self.assertEqual(b"\x01\x02\x03\x04\x05", bytes(self._instance))

    def test_magic_getitem(self) -> None:
        self.assertEqual(3, self._instance[2])
        assert_array_equal([1, 2], self._instance[:2])

    def test_magic_iter(self) -> None:
        for ref, res in zip(range(1, 6), self._instance):
            self.assertEqual(ref, res)

    def test_magic_len(self) -> None:
        self.assertEqual(5, len(self._instance))

    @property
    def _instance(self) -> Field:
        return Field(name="test", struct=Struct(stop=5))


class TestBytesStorageABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance,
            content=b"",
            name="test_storage",
        )

    def test_encode_extract(self) -> None:
        obj = self._instance.encode(bytes(range(20)))

        validate_object(
            self,
            obj,
            content=bytes(range(20)),
            wo_attrs=["name"],
        )

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

    def test_encode_extract_without_infinite(self) -> None:
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

    def test_encode_set(self) -> None:

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

    def test_encode_change_field_content(self) -> None:
        obj = self._instance.encode(
                second=[1, 2],
                fourth=(3, 4, 5),
                fifth=6,
            )
        self.assertEqual(b"\xfa" + bytes(range(1, 7)), obj.content)

        obj.encode(first=b"\x00")
        self.assertEqual(bytes(range(7)), obj.content)

    def test_encode_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            self._instance.encode(b"\x01", first=21)
        self.assertEqual(
            "takes a message or fields (both given)", exc.exception.args[0]
        )

        with self.assertRaises(TypeError) as exc:
            self._instance.encode(b"")
        self.assertEqual("message is empty", exc.exception.args[0])

    def test_encode_exc_wrong_content(self) -> None:
        with self.assertRaises(ValueError) as exc:
            self._instance.encode(
                second=[1, 2],
                fourth=(3, 4, 5),
                fifth=[2, 3],
            )
        self.assertEqual("'02 03' is not correct for 'fifth'", exc.exception.args[0])

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

    @property
    def _instance(self) -> Storage:
        return Storage("test_storage", dict(
            first=Struct(start=0, default=b"\xfa", stop=1),
            second=Struct(start=1, bytes_expected=2),
            third=Struct(start=3, stop=-4),
            fourth=Struct(start=-4, stop=-1),
            fifth=Struct(start=-1, stop=None),
        ))


class TestBytesStoragePatternABC(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_write(self) -> None:
        path = DATA_DIR / "test_write.ini"
        self._instance.write(path)

        i_line = 0
        with open(path, "r") as file:
            for ref, res in zip(
                    [
                        "[test]",
                        r"test = \dct(typename,base,name,test,val,\tpl(11))",
                        r"first = \dct(typename,base,int,3,list,\lst(2,3,4))",
                        r"second = \dct(typename,base,boolean,True)",
                        r"third = \dct(typename,base,dict,\dct(0,1,2,3))",
                    ],
                    file.read().split("\n")
            ):
                i_line += 1
                with self.subTest(test="new", line=i_line):
                    self.assertEqual(ref, res)
        self.assertEqual(5, i_line)

        StoragePattern(typename="base", name="test", val=(11,)).configure(
            first=FieldPattern(typename="base", int=11),
        ).write(path)

        i_line = 0
        with open(path, "r") as file:
            for ref, res in zip(
                    [
                        "[test]",
                        r"test = \dct(typename,base,name,test,val,\tpl(11))",
                        r"first = \dct(typename,base,int,11)",
                    ],
                    file.read().split("\n")
            ):
                i_line += 1
                with self.subTest(test="rewrite", line=i_line):
                    self.assertEqual(ref, res)
        self.assertEqual(3, i_line)

    def test_write_exc_not_configured(self) -> None:
        with self.assertRaises(NotConfiguredYet) as exc:
            StoragePattern(typename="base", name="test", val=(11,)).write(
                DATA_DIR / "test.ini"
            )
        self.assertEqual(
            "StoragePattern not configured yet", exc.exception.args[0]
        )

    def test_write_read(self) -> None:
        path = DATA_DIR / "test_write_read.ini"
        ref = self._instance
        ref.write(path)
        res = StoragePattern.read(path, "test")

        self.assertIsNot(ref, res)
        self.assertEqual(ref, res)
        self.assertEqual(ref._sub_p, res._sub_p)

    def test_read_many_keys(self) -> None:
        with self.assertRaises(TypeError) as exc:
            StoragePattern.read(DATA_DIR, "key_1", "key_2")
        self.assertEqual("given 2 keys, expect one", exc.exception.args[0])

    @property
    def _instance(self) -> StoragePattern:
        return StoragePattern(typename="base", name="test", val=(11,)).configure(
            first=FieldPattern(typename="base", int=3, list=[2, 3, 4]),
            second=FieldPattern(typename="base", boolean=True),
            third=FieldPattern(
                typename="base", dict={0: 1, 2: 3}
            )
        )


# todo: fix tests for ContinuousBytesStorage
# class TestContinuousBytesStoragePatternABC(unittest.TestCase):
#
#     def test_get_continuous(self) -> None:
#         # obj = self._instance._get_continuous(
#         #     True,
#         #
#         # )
#         ...
#
#     def test_get_continuous_exc(self) -> None:
#         with self.subTest(test="repeated kwarg"):
#             with self.assertRaises(SyntaxError) as exc:
#                 self._instance._get_continuous(
#                     False,
#                     {"name": "test"},
#                     {}
#                 )
#             self.assertEqual(
#                 "keyword argument(s) repeated: name", exc.exception.args[0]
#             )
#
#         with self.subTest(test="with 'fields'"):
#             with self.assertRaises(TypeError) as exc:
#                 self._instance._get_continuous(
#                     False,
#                     {"fields": {}},
#                     {}
#                 )
#             self.assertEqual(
#                 "'fields' is an invalid keyword argument for continuous "
#                 "storage",
#                 exc.exception.args[0],
#             )
#
#     @property
#     def _instance(self) -> ContinuousStoragePattern:
#         def get_pattern(**kwargs) -> FieldPattern:
#             return FieldPattern(typename="base", **kwargs)
#
#         return ContinuousStoragePattern(
#             typename="base",
#             name="test",
#         ).configure(
#             first=get_pattern(default=b"\xfa"),
#             second=get_pattern(),
#             third=get_pattern(is_infinite=True),
#             fourth=get_pattern(),
#             fifth=get_pattern(),
#
#         )

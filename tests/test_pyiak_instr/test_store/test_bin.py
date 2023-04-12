import shutil
import unittest
from typing import Any

import numpy as np

from src.pyiak_instr.core import Code
from src.pyiak_instr.store import (
    BytesFieldStruct,
    ContinuousBytesStorage,
    BytesFieldPattern,
    BytesStoragePattern,
)
from src.pyiak_instr.exceptions import NotConfiguredYet

from ..data_bin import (
    get_cbs_one,
    get_cbs_example,
    get_cbs_one_infinite,
    get_cbs_first_infinite,
    get_cbs_middle_infinite,
    get_cbs_last_infinite,
)
from ..env import get_local_test_data_dir
from ...utils import validate_object, compare_values

TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestBytesFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            BytesFieldStruct(
                start=4,
                fmt=Code.U32,
                bytes_expected=-100,
            ),
            bytes_expected=0,
            default=b"",
            fmt=Code.U32,
            has_default=False,
            is_floating=True,
            order=Code.BIG_ENDIAN,
            slice_=slice(4, None),
            start=4,
            stop=None,
            word_bytesize=4,
            words_expected=0
        )

    def test_decode(self) -> None:
        obj = BytesFieldStruct(
            start=4,
            fmt=Code.U32,
            bytes_expected=-100,
        )
        cases = (
            (b"\x00\x00\x00\x01", 1),
            (b"\x00\x00\x00\x01\x00\x00\x00\x02", [1, 2]),
            (b"\x00\x00\x00\x01\x00\x00\x00\x22", np.array([1, 0x22])),
        )
        for i_case, (data, ref) in enumerate(cases):
            with self.subTest(test=i_case):
                compare_values(self, ref, obj.decode(data))

    def test_encode(self) -> None:
        obj = BytesFieldStruct(
            start=4,
            fmt=Code.U32,
            bytes_expected=-100,
            order=Code.BIG_ENDIAN,
        )
        cases = (
            (1, b"\x00\x00\x00\x01"),
            ([1, 2], b"\x00\x00\x00\x01\x00\x00\x00\x02"),
            (np.array([1, 0x22]), b"\x00\x00\x00\x01\x00\x00\x00\x22"),
        )
        for i_case, (data, ref) in enumerate(cases):
            with self.subTest(test=i_case):
                compare_values(self, ref, obj.encode(data))

    def test_validate(self) -> None:
        obj = BytesFieldStruct(
            start=0,
            fmt=Code.I16,
            bytes_expected=4,
            order=Code.LITTLE_ENDIAN,
        )
        with self.subTest(test="finite True"):
            self.assertTrue(obj.validate(b"\x02\x04\x00\x00"))
        with self.subTest(test="finite False"):
            self.assertFalse(obj.validate(b"\x02\x04\x00\x00\x01"))

        obj = BytesFieldStruct(
            start=0,
            fmt=Code.I16,
            bytes_expected=-1,
            order=Code.LITTLE_ENDIAN,
        )
        with self.subTest(test="infinite True"):
            self.assertTrue(obj.validate(b"\x02\x04\x00\x00"))
        with self.subTest(test="infinite False"):
            self.assertFalse(obj.validate(b"\x02\x04\x00"))

    def test_stop_after_infinite(self) -> None:
        data = (
            (BytesFieldStruct(
                start=-4,
                fmt=Code.I16,
                bytes_expected=2,
                order=Code.LITTLE_ENDIAN,
            ), -2),
            (BytesFieldStruct(
                start=-2,
                fmt=Code.I16,
                bytes_expected=2,
                order=Code.LITTLE_ENDIAN,
            ), None)
        )

        for i, (obj, ref) in enumerate(data):
            with self.subTest(test=i):
                self.assertEqual(ref, obj.stop)


class TestBytesField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._get_cbs()["f0"],
            bytes_count=0,
            content=b"",
            name="f0",
            words_count=0,
            wo_attrs=["struct"],
        )

    def test_decode(self) -> None:
        compare_values(self, [], self._get_cbs()["f0"].decode())

    @staticmethod
    def _get_cbs() -> ContinuousBytesStorage:
        return ContinuousBytesStorage(
                "cbs", {
                    "f0": BytesFieldStruct(
                        start=0,
                        fmt=Code.U32,
                        bytes_expected=-1,
                        order=Code.BIG_ENDIAN,
                    )
                }
            )


class TestContinuousBytesStorage(unittest.TestCase):

    def test_init(self) -> None:
        obj = self._get_cbs()
        ref_data = dict(
            f0=dict(
                bytes_count=0,
                content=b"",
                name="f0",
                words_count=0,
            ),
        )

        validate_object(self, obj, content=b"", name="cbs")
        for name, ref in ref_data.items():
            validate_object(self, obj[name], **ref, wo_attrs=["struct"])

    def test_init_exc(self) -> None:
        with self.assertRaises(TypeError)as exc:
            ContinuousBytesStorage("", {"f0": {}})
        self.assertEqual(
            "invalid type of 'f0': <class 'dict'>", exc.exception.args[0]
        )

    def test_encode_extract(self) -> None:

        def get_storage_pars(
                content: bytes,
                name: str,
        ) -> dict[str, Any]:
            return dict(
                content=content,
                name=name,
            )

        def get_field_pars(
                content: bytes,
                words_count: int,
        ) -> dict[str, Any]:
            return dict(
                bytes_count=len(content),
                content=content,
                words_count=words_count,
                wo_attrs=["struct", "name"],
            )

        data = dict(
            one=dict(
                obj=get_cbs_one(),
                data=b"\x00\x01\xff\xff",
                validate_storage=get_storage_pars(
                    b"\x00\x01\xff\xff", "cbs_one"
                ),
                validate_fields=dict(
                    f0=get_field_pars(b"\x00\x01\xff\xff", 2),
                ),
                decode=dict(
                    f0=[1, -1],
                ),
            ),
            one_infinite=dict(
                obj=get_cbs_one_infinite(),
                data=b"\xef\x01\xff",
                validate_storage=get_storage_pars(
                    b"\xef\x01\xff", "cbs_one_infinite"
                ),
                validate_fields=dict(
                    f0=get_field_pars(b"\xef\x01\xff", 3),
                ),
                decode=dict(
                    f0=[-17, 1, -1],
                ),
            ),
            first_infinite=dict(
                obj=get_cbs_first_infinite(),
                data=b"\xef\x01\xff\xff\x0f\xab\xdd",
                validate_storage=get_storage_pars(
                    b"\xef\x01\xff\xff\x0f\xab\xdd", "cbs_first_infinite"
                ),
                validate_fields=dict(
                    f0=get_field_pars(b"\xef\x01\xff", 3),
                    f1=get_field_pars(b"\xff\x0f", 1),
                    f2=get_field_pars(b"\xab\xdd", 2),
                ),
                decode=dict(
                    f0=[-17, 1, -1],
                    f1=[0xFF0F],
                    f2=[0xAB, 0xDD],
                ),
            ),
            middle_infinite=dict(
                obj=get_cbs_middle_infinite(),
                data=b"\xef\x01\xff\xff\x01\x02\x03\x04\xab\xdd",
                validate_storage=get_storage_pars(
                    b"\xef\x01\xff\xff\x01\x02\x03\x04\xab\xdd",
                    "cbs_middle_infinite",
                ),
                validate_fields=dict(
                    f0=get_field_pars(b"\xef\x01\xff\xff", 2),
                    f1=get_field_pars(b"\x01\x02\x03\x04", 2),
                    f2=get_field_pars(b"\xab\xdd", 2),
                ),
                decode=dict(
                    f0=[0x1EF, 0xFFFF],
                    f1=[0x102, 0x304],
                    f2=[0xAB, 0xDD],
                ),
            ),
            last_infinite=dict(
                obj=get_cbs_last_infinite(),
                data=b"\xab\xcd\x00\x00\x01\x02\x03\x04\x00\x00",
                validate_storage=get_storage_pars(
                    b"\xab\xcd\x00\x00\x01\x02\x03\x04\x00\x00",
                    "cbs_last_infinite",
                ),
                validate_fields=dict(
                    f0=get_field_pars(b"\xab\xcd", 1),
                    f1=get_field_pars(
                        b"\x00\x00\x01\x02\x03\x04\x00\x00", 2
                    ),
                ),
                decode=dict(
                    f0=[0xCDAB],
                    f1=[0x102, 0x3040000],
                ),
            ),
        )

        for short_name, comp in data.items():
            with self.subTest(test=short_name):
                obj = comp["obj"]
                comp_storage = comp["validate_storage"]
                comp_fields = comp["validate_fields"]
                comp_decode = comp["decode"]

                obj.encode(comp["data"])
                validate_object(self, obj, **comp_storage)
                for field in obj:
                    validate_object(self, field, **comp_fields[field.name])

                with self.subTest(sub_test="decode"):
                    self.assertListEqual(
                        list(comp_decode), [f.name for f in obj]
                    )
                    for f_name, decoded in obj.decode().items():
                        with self.subTest(field=f_name):
                            compare_values(self, comp_decode[f_name], decoded)

    def test_encode_replace(self) -> None:
        obj = get_cbs_example()
        obj.encode(f0=1, f1=[2, 3], f3=[4, 5], f4=6)
        self.assertEqual(b"\x00\x01\x02\x03\x04\x05\x06", obj.content)
        obj.encode(f2=b"\xff\xfe\xfd")
        self.assertEqual(
            b"\x00\x01\x02\x03\xff\xfe\xfd\x04\x05\x06",
            obj.content,
        )
        obj.encode(f0=32)
        self.assertEqual(
            b"\x00\x20\x02\x03\xff\xfe\xfd\x04\x05\x06",
            obj.content,
        )
        obj.encode(f2=b"new content")
        self.assertEqual(
            b"\x00\x20\x02\x03new content\x04\x05\x06",
            obj.content,
        )

    def test_encode_exc(self) -> None:
        with self.subTest(exception="missing or extra"):
            with self.assertRaises(AttributeError) as exc:
                get_cbs_example().encode(f0=1, f1=2, f4=5, f8=1)
            self.assertEqual(
                "missing or extra fields were found: 'f3', 'f8'",
                exc.exception.args[0],
            )

        with self.subTest(exception="invalid new content"):
            with self.assertRaises(ValueError) as exc:
                get_cbs_example().encode(f0=1, f1=[2, 3, 4], f3=1, f4=5)
            self.assertEqual(
                "'02 03 04' is not correct for 'f1'", exc.exception.args[0]
            )

    def test_magic_contains(self) -> None:
        self.assertIn("f0", self._get_cbs())

    def test_magic_getitem(self) -> None:
        self.assertEqual("f0", self._get_cbs()["f0"].name)

    def test_magic_iter(self) -> None:
        for ref, res in zip(("f0",), self._get_cbs()):
            with self.subTest(ref=ref):
                self.assertEqual(ref, res.name)

    @staticmethod
    def _get_cbs() -> ContinuousBytesStorage:
        return ContinuousBytesStorage(
            "cbs", dict(
                f0=BytesFieldStruct(
                    start=0,
                    fmt=Code.U32,
                    bytes_expected=4,
                ),
            )
        )


# note: comment editable pattern tests
class TestBytesFieldPattern(unittest.TestCase):

    def test_get(self) -> None:
        pattern = BytesFieldPattern(
            fmt=Code.U8,
            order=Code.LITTLE_ENDIAN,
            bytes_expected=4,
        )
        validate_object(
            self,
            pattern.get(start=4),
            start=4,
            fmt=Code.U8,
            order=Code.LITTLE_ENDIAN,
            bytes_expected=4,
            check_attrs=False,
        )

    def test_get_updated(self) -> None:
        pattern = BytesFieldPattern(
            fmt=Code.U8,
            bytes_expected=4,
        )
        with self.assertRaises(SyntaxError) as exc:
            pattern.get(start=0, bytes_expected=1)
        self.assertEqual(
            "keyword argument(s) repeated: bytes_expected",
            exc.exception.args[0],
        )

        validate_object(
            self,
            pattern.get(changes_allowed=True, start=0, bytes_expected=1),
            start=0,
            bytes_expected=1,
            check_attrs=False,
        )

    def test_magic_contains(self) -> None:
        self.assertIn("a", self._get_pattern())

    def test_magic_getitem(self) -> None:
        self.assertEqual(1, self._get_pattern()["a"])

    @staticmethod
    def _get_pattern() -> BytesFieldPattern:
        return BytesFieldPattern(
            a=1,
            b=[],
            c={},
            d="string"
        )


class TestBytesStoragePattern(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_DIR.parent)

    def test_init(self) -> None:
        validate_object(
            self,
            BytesStoragePattern(typename="continuous", name="cbs", kwarg="None"),
            name="cbs",
            typename="continuous",
        )

    def test_get_continuous(self) -> None:
        data = b"\xaa\x55\xab\xcd\x11\x22\x33\x44\x55\xdc\xbb\x99"
        ref = dict(
            validate_storage=dict(
                content=b"\xaa\x55\xab\xcd\x11\x22\x33\x44\x55\xdc\xbb\x99",
                name="cbs_example"
            ),
            validate_fields=dict(
                f0=dict(
                    content=b"\xaa\x55",
                    words_count=1,
                ),
                f1=dict(
                    content=b"\xab\xcd",
                    words_count=2,
                ),
                f2=dict(
                    content=b"\x11\x22\x33\x44\x55",
                    words_count=5,
                ),
                f3=dict(
                    content=b"\xdc\xbb",
                    words_count=2,
                ),
                f4=dict(
                    content=b"\x99",
                    words_count=1,
                ),
            ),
            field_slices=dict(
                f0=slice(0, 2),
                f1=slice(2, 4),
                f2=slice(4, -3),
                f3=slice(-3, -1),
                f4=slice(-1, None),
            ),
            decode=dict(
                f0=[0xAA55],
                f1=[0xAB, 0xCD],
                f2=[0x11, 0x22, 0x33, 0x44, 0x55],
                f3=[0xDC, 0xBB],
                f4=[-103],
            ),
        )

        pattern = self._get_example_pattern()
        res = pattern.get()
        res.encode(data)

        ref_storage = ref["validate_storage"]
        ref_fields = ref["validate_fields"]
        ref_slice = ref["field_slices"]
        ref_decode = ref["decode"]

        validate_object(self, res, **ref_storage)
        for field in res:
            validate_object(
                self, field, **ref_fields[field.name], check_attrs=False,
            )

        with self.subTest(test="slice"):
            self.assertListEqual(list(ref_slice), [f.name for f in res])
            for parser in res:
                with self.subTest(field=parser.name):
                    compare_values(
                        self, ref_slice[parser.name], parser.struct.slice_
                    )

        with self.subTest(test="decode"):
            self.assertListEqual(list(ref_decode), [f.name for f in res])
            for f_name, decoded in res.decode().items():
                with self.subTest(field=f_name):
                    compare_values(self, ref_decode[f_name], decoded)

    def test_get_exc(self) -> None:
        with self.assertRaises(NotConfiguredYet) as exc:
            BytesStoragePattern(typename="continuous", name="test").get()
        self.assertEqual(
            "BytesStoragePattern not configured yet", exc.exception.args[0]
        )

    def test_read_write_config(self) -> None:
        path = TEST_DATA_DIR / "test.ini"
        ref = self._get_example_pattern()
        ref.write(path)
        res = BytesStoragePattern.read(path, "cbs_example")

        with self.subTest(test="storage"):
            self.assertDictEqual(ref._kw, res._kw)

        for (name, rf), (_, rs) in zip(ref._sub_p.items(), res._sub_p.items()):
            with self.subTest(test=name):
                for (par, rf_kw), rs_kw in zip(
                        rf._kw.items(), rs._kw.values()
                ):
                    with self.subTest(parameter=par):
                        self.assertEqual(rf_kw, rs_kw)
                        self.assertIsInstance(rs_kw, type(rf_kw))
        res.get()

    @staticmethod
    def _get_example_pattern() -> BytesStoragePattern:
        pattern = BytesStoragePattern(
            name="cbs_example", typename="continuous"
        )
        pattern.configure(
            f0=BytesFieldPattern(
                fmt=Code.U16,
                bytes_expected=2,
                order=Code.BIG_ENDIAN,
            ),
            f1=BytesFieldPattern(
                fmt=Code.U8,
                bytes_expected=2,
            ),
            f2=BytesFieldPattern(
                fmt=Code.I8,
                bytes_expected=-1,
            ),
            f3=BytesFieldPattern(
                fmt=Code.U8,
                bytes_expected=2,
            ),
            f4=BytesFieldPattern(
                fmt=Code.I8,
                bytes_expected=1,
            )
        )
        return pattern

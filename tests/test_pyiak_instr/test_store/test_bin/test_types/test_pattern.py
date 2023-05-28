import unittest
import shutil
from typing import Any

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.types import SubPatternAdditions

from .....utils import compare_objects, validate_object
from ....env import get_local_test_data_dir
from .ti import (
    TIFieldStruct,
    TIFieldStructPattern,
    TIStorageStruct,
    TIStorageStructPattern,
    TIStoragePattern,
    TIContinuousStorageStructPattern,
)


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestBytesFieldStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIFieldStructPattern(typename="basic"),
            typename="basic",
            is_dynamic=True,
            size=0,
        )

    def test_get(self) -> None:
        validate_object(
            self,
            TIFieldStructPattern(
                typename="basic", bytes_expected=4, fmt=Code.U16
            ).get(),
            has_default=False,
            name="",
            stop=4,
            start=0,
            words_expected=2,
            order=Code.BIG_ENDIAN,
            word_bytesize=2,
            bytes_expected=4,
            is_dynamic=False,
            default=b"",
            slice_=slice(0, 4),
            fmt=Code.U16,
            wo_attrs=["encoder"],
        )

    def test_size(self) -> None:
        def __pattern(**parameters: Any):
            return TIFieldStructPattern(typename="basic", **parameters)

        for i, (ref, obj) in enumerate((
            (0, __pattern(bytes_expected=0)),
            (2, __pattern(bytes_expected=2)),
            (0, __pattern()),
            (0, __pattern(stop=None)),
            (0, __pattern(start=2, stop=None)),
            (4, __pattern(stop=4)),
            (2, __pattern(start=2, stop=4)),
            (2, __pattern(start=-4, stop=-2)),
            (4, __pattern(start=-4, stop=None)),
            (4, __pattern(start=-4)),
            (0, __pattern(start=4, stop=-2)),
        )):
            with self.subTest(case=i):
                self.assertEqual(ref, obj.size)


class TestBytesStorageStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStorageStructPattern(typename="basic", name="test"),
            typename="basic",
            sub_pattern_names=[],
        )

    def test_get(self) -> None:
        storage = TIStorageStructPattern(
            typename="basic", name="test"
        ).configure(
                f0=TIFieldStructPattern(typename="basic", bytes_expected=3)
            ).get()

        validate_object(
            self,
            storage,
            dynamic_field_name="",
            is_dynamic=False,
            minimum_size=3,
            name="test",
            fields={},
        )
        validate_object(
            self,
            storage["f0"],
            has_default=False,
            name="f0",
            stop=3,
            start=0,
            words_expected=3,
            order=Code.BIG_ENDIAN,
            word_bytesize=1,
            bytes_expected=3,
            is_dynamic=False,
            default=b"",
            slice_=slice(0, 3),
            fmt=Code.U8,
            wo_attrs=["encoder"],
        )


class TestBytesStoragePatternABC(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_init(self) -> None:
        validate_object(
            self,
            TIStoragePattern(typename="basic", name="test"),
            typename="basic",
            sub_pattern_names=[],
        )

    def test_get(self) -> None:
        storage = TIStoragePattern(typename="basic").configure(
            test=TIStorageStructPattern(typename="basic").configure(
                f0=TIFieldStructPattern(typename="basic", bytes_expected=2)
            )
        )
        res = storage.get(sub_additions=SubPatternAdditions().set_next_additions(
            test=SubPatternAdditions().update_additions("f0", fmt=Code.U16)
        ))

        validate_object(
            self,
            res,
            has_pattern=True,
            wo_attrs=["struct", "pattern"],
        )
        validate_object(
            self,
            res.struct,
            dynamic_field_name="",
            is_dynamic=False,
            minimum_size=2,
            name="test",
            fields={},
        )
        validate_object(
            self,
            res.struct["f0"],
            has_default=False,
            name="f0",
            stop=2,
            start=0,
            words_expected=1,
            order=Code.BIG_ENDIAN,
            word_bytesize=2,
            bytes_expected=2,
            is_dynamic=False,
            default=b"",
            slice_=slice(0, 2),
            fmt=Code.U16,
            wo_attrs=["encoder"],
        )

    def test_configure_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIStoragePattern(typename="basic").configure(
                s0=TIStorageStructPattern(typename="basic"),
                s1=TIStorageStructPattern(typename="basic"),
            )
        self.assertEqual(
            "only one storage pattern allowed, got 2", exc.exception.args[0]
        )

    def test_write(self) -> None:
        path = TEST_DATA_DIR / "test_write.ini"
        self._instance().write(path)

        ref = [
            "[test]",
            r"_ = \dct(typename,basic,val,\str(33))",
            r"test = \dct(typename,basic,val,\tpl(11))",
            r"first = \dct(typename,basic,bytes_expected,0,int,3,"
            r"list,\lst(2,3,4))",
            r"second = \dct(typename,basic,bytes_expected,0,boolean,True)",
            r"third = \dct(typename,basic,bytes_expected,0,"
            r"dict,\dct(0,1,2,3))",
        ]
        i_line = 0
        with open(path, "r") as file:
            for rf, rs in zip(ref, file.read().split("\n")):
                i_line += 1
                with self.subTest(test="new", line=i_line):
                    self.assertEqual(rf, rs)
        self.assertEqual(len(ref), i_line)

        TIStoragePattern(typename="basic").configure(
            test=TIStorageStructPattern(
                typename="basic", in33=0
            ).configure(
                f0=TIFieldStructPattern(typename="basic", bytes_expected=33)
            ),
        ).write(path)

        ref = [
            "[test]",
            r"_ = \dct(typename,basic)",
            r"test = \dct(typename,basic,in33,0)",
            r"f0 = \dct(typename,basic,bytes_expected,33)",
        ]
        i_line = 0
        with open(path, "r") as file:
            for rf, rs in zip(ref, file.read().split("\n")):
                i_line += 1
                with self.subTest(test="rewrite", line=i_line):
                    self.assertEqual(rf, rs)
        self.assertEqual(len(ref), i_line)

    def test_write_exc_not_configured(self) -> None:
        with self.assertRaises(NotConfiguredYet) as exc:
            TIStoragePattern(typename="basic", name="test", val=(11,)).write(
                TEST_DATA_DIR / "test.ini"
            )
        self.assertEqual(
            "TIStoragePattern not configured yet", exc.exception.args[0]
        )

    def test_write_read(self) -> None:
        path = TEST_DATA_DIR / "test_write_read.ini"
        ref = self._instance()
        ref.write(path)
        res = TIStoragePattern.read(path, "test")

        self.assertIsNot(ref, res)
        self.assertEqual(ref, res)
        self.assertEqual(ref._sub_p, res._sub_p)
        self.assertEqual(ref._sub_p["test"]._sub_p, res._sub_p["test"]._sub_p)

    def test_read_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIStoragePattern.read(TEST_DATA_DIR, "key_1", "key_2")
        self.assertEqual(
            "TIStoragePattern takes only 1 argument (2 given)",
            exc.exception.args[0],
        )

    @staticmethod
    def _instance() -> TIStoragePattern:
        struct_pattern = TIStorageStructPattern(
            typename="basic", val=(11,)
        )
        struct_pattern.configure(
            first=TIFieldStructPattern(
                typename="basic", bytes_expected=0, int=3, list=[2, 3, 4]
            ),
            second=TIFieldStructPattern(
                typename="basic", bytes_expected=0, boolean=True
            ),
            third=TIFieldStructPattern(
                typename="basic", bytes_expected=0, dict={0: 1, 2: 3}
            )
        )
        pattern = TIStoragePattern(typename="basic", val="33")
        pattern.configure(test=struct_pattern)
        return pattern


class TestContinuousBytesStorageStructPatternABC(unittest.TestCase):

    def test_get(self) -> None:
        pattern = TIContinuousStorageStructPattern(
            typename="basic", name="test"
        ).configure(
            f0=TIFieldStructPattern(typename="basic", bytes_expected=4),
            f1=TIFieldStructPattern(typename="basic", bytes_expected=2),
            f2=TIFieldStructPattern(typename="basic", bytes_expected=0),
            f3=TIFieldStructPattern(typename="basic", bytes_expected=3),
            f4=TIFieldStructPattern(typename="basic", bytes_expected=4),
        )

        ref = TIStorageStruct(name="test", fields=dict(
                f0=TIFieldStruct(name="f0", start=0, bytes_expected=4),
                f1=TIFieldStruct(name="f1", start=4, bytes_expected=2),
                f2=TIFieldStruct(name="f2", start=6, stop=-7),
                f3=TIFieldStruct(name="f3", start=-7, bytes_expected=3),
                f4=TIFieldStruct(name="f4", start=-4, bytes_expected=4),
        ))
        res = pattern.get()
        compare_objects(self, ref, res)
        for ref_field, res_field in zip(ref, res):
            with self.subTest(field=ref_field.name):
                compare_objects(self, ref_field, res_field)

    def test__modify_sub_additions(self) -> None:
        with self.subTest(test="dyn in middle"):
            for_subs = SubPatternAdditions()
            TIContinuousStorageStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldStructPattern(name="f0", typename="basic", bytes_expected=4),
                f1=TIFieldStructPattern(name="f1", typename="basic", bytes_expected=2),
                f2=TIFieldStructPattern(name="f2", typename="basic", bytes_expected=0),
                f3=TIFieldStructPattern(name="f3", typename="basic", bytes_expected=3),
                f4=TIFieldStructPattern(name="f4", typename="basic", bytes_expected=4),
            )._modify_sub_additions(for_subs)
            self.assertDictEqual(
                dict(
                    f0=dict(name="f0", start=0),
                    f1=dict(name="f1", start=4),
                    f2=dict(name="f2", start=6, stop=-7),
                    f3=dict(name="f3", start=-7),
                    f4=dict(name="f4", start=-4),
                ), for_subs.additions
            )

        with self.subTest(test="last dyn"):
            for_subs = SubPatternAdditions().update_additions("f2", req=1)
            TIContinuousStorageStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldStructPattern(name="f0", typename="basic", bytes_expected=4),
                f1=TIFieldStructPattern(name="f1", typename="basic", bytes_expected=2),
                f2=TIFieldStructPattern(name="f2", typename="basic", bytes_expected=0),
            )._modify_sub_additions(for_subs)
            self.assertDictEqual(
                dict(
                    f0=dict(name="f0", start=0),
                    f1=dict(name="f1", start=4),
                    f2=dict(name="f2", start=6, stop=None, req=1),
                ),
                for_subs.additions
            )

        with self.subTest(test="only dyn"):
            for_subs = SubPatternAdditions()
            TIContinuousStorageStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldStructPattern(name="f0", typename="basic", bytes_expected=0),
            )._modify_sub_additions(for_subs)
            self.assertDictEqual(
                dict(f0=dict(name="f0", start=0, stop=None)),
                for_subs.additions,
            )

        with self.subTest(test="without dyn"):
            for_subs = SubPatternAdditions()
            TIContinuousStorageStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldStructPattern(name="f0", typename="basic", bytes_expected=4),
                f1=TIFieldStructPattern(name="f1", typename="basic", bytes_expected=2),
                f2=TIFieldStructPattern(name="f2", typename="basic", bytes_expected=3),
            )._modify_sub_additions(for_subs)
            self.assertDictEqual(
                dict(
                    f0=dict(name="f0", start=0),
                    f1=dict(name="f1", start=4),
                    f2=dict(name="f2", start=6),
                ),
                for_subs.additions,
            )

    def test__modify_sub_additions_exc(self) -> None:
        with self.subTest(test="two dynamic field"):
            with self.assertRaises(TypeError) as exc:
                TIContinuousStorageStructPattern(
                    typename="basic", name="test",
                ).configure(
                    f0=TIFieldStructPattern(name="f0", typename="basic", bytes_expected=0),
                    f1=TIFieldStructPattern(name="f1", typename="basic", bytes_expected=0),
                )._modify_sub_additions(SubPatternAdditions())
            self.assertEqual(
                "two dynamic field not allowed", exc.exception.args[0]
            )

import unittest
import shutil
from typing import Any

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.types import Additions

from tests.utils import compare_objects, validate_object
from tests.test_pyiak_instr.env import get_local_test_data_dir, remove_test_data_dir

from tests.pyiak_instr_ti.store import (
    TIField,
    TIFieldPattern,
    TIStruct,
    TIStructPattern,
    TIContainerPattern,
    TIContinuousStructPattern,
)


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestFieldPattern(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIFieldPattern(typename="basic"),
            typename="basic",
            is_dynamic=True,
            bytesize=0,
            wo_attrs=["additions"],
        )

    def test_get(self) -> None:
        validate_object(
            self,
            TIFieldPattern(
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
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_bytesize(self) -> None:
        def __pattern(**parameters: Any):
            return TIFieldPattern(typename="basic", **parameters)

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
                self.assertEqual(ref, obj.bytesize)


class TestStructPattern(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStructPattern(typename="basic", name="test"),
            typename="basic",
            sub_pattern_names=[],
            wo_attrs=["additions"],
        )

    def test_get(self) -> None:
        storage = TIStructPattern(
            typename="basic", name="test"
        ).configure(
                f0=TIFieldPattern(typename="basic", bytes_expected=3)
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
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestBytesStoragePatternABC(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_init(self) -> None:
        validate_object(
            self,
            TIContainerPattern(typename="basic", name="test"),
            typename="basic",
            sub_pattern_names=[],
            wo_attrs=["additions"],
        )

    def test_get(self) -> None:
        additions = Additions(
            lower={"test": Additions(
                lower={"f0": Additions(current={"fmt": Code.U16})}
            )}
        )
        storage = TIContainerPattern(typename="basic").configure(
            test=TIStructPattern(typename="basic").configure(
                f0=TIFieldPattern(typename="basic", bytes_expected=2)
            )
        )
        res = storage.get(additions=additions)

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
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_configure_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIContainerPattern(typename="basic").configure(
                s0=TIStructPattern(typename="basic"),
                s1=TIStructPattern(typename="basic"),
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

        TIContainerPattern(typename="basic").configure(
            test=TIStructPattern(
                typename="basic", in33=0
            ).configure(
                f0=TIFieldPattern(typename="basic", bytes_expected=33)
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
            TIContainerPattern(typename="basic", name="test", val=(11,)).write(
                TEST_DATA_DIR / "test.ini"
            )
        self.assertEqual(
            "TIContainerPattern not configured yet", exc.exception.args[0]
        )

    def test_write_read(self) -> None:
        path = TEST_DATA_DIR / "test_write_read.ini"
        ref = self._instance()
        ref.write(path)
        res = TIContainerPattern.read(path, "test")

        self.assertIsNot(ref, res)
        self.assertDictEqual(ref.__init_kwargs__(), res.__init_kwargs__())
        self.assertEqual(
            ref._sub_p["test"].__init_kwargs__(),
            res._sub_p["test"].__init_kwargs__(),
        )
        for name in ref._sub_p["test"]._sub_p:
            self.assertEqual(
                ref._sub_p["test"]._sub_p[name].__init_kwargs__(),
                res._sub_p["test"]._sub_p[name].__init_kwargs__(),
            )

    def test_read_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIContainerPattern.read(TEST_DATA_DIR, "")
        self.assertEqual("empty name not allowed", exc.exception.args[0])

    @staticmethod
    def _instance() -> TIContainerPattern:
        struct_pattern = TIStructPattern(
            typename="basic", val=(11,)
        )
        struct_pattern.configure(
            first=TIFieldPattern(
                typename="basic", bytes_expected=0, int=3, list=[2, 3, 4]
            ),
            second=TIFieldPattern(
                typename="basic", bytes_expected=0, boolean=True
            ),
            third=TIFieldPattern(
                typename="basic", bytes_expected=0, dict={0: 1, 2: 3}
            )
        )
        pattern = TIContainerPattern(typename="basic", val="33")
        pattern.configure(test=struct_pattern)
        return pattern


class TestContinuousBytesStorageStructPatternABC(unittest.TestCase):

    def test_get(self) -> None:
        pattern = TIContinuousStructPattern(
            typename="basic", name="test"
        ).configure(
            f0=TIFieldPattern(typename="basic", bytes_expected=4),
            f1=TIFieldPattern(typename="basic", bytes_expected=2),
            f2=TIFieldPattern(typename="basic", bytes_expected=0),
            f3=TIFieldPattern(typename="basic", bytes_expected=3),
            f4=TIFieldPattern(typename="basic", bytes_expected=4),
        )

        ref = TIStruct(name="test", fields=dict(
                f0=TIField(name="f0", start=0, bytes_expected=4),
                f1=TIField(name="f1", start=4, bytes_expected=2),
                f2=TIField(name="f2", start=6, stop=-7),
                f3=TIField(name="f3", start=-7, bytes_expected=3),
                f4=TIField(name="f4", start=-4, bytes_expected=4),
        ))
        res = pattern.get()
        compare_objects(self, ref, res)
        for ref_field, res_field in zip(ref, res):
            with self.subTest(field=ref_field.name):
                compare_objects(self, ref_field, res_field, wo_attrs=["encoder"])

    def test__modify_additions(self) -> None:

        def compare_subs(exp: dict[str, Any], act: Additions):
            for name, kw in exp.items():
                self.assertDictEqual(kw, act.lower(name).current)

        with self.subTest(test="dyn in middle"):
            adds = Additions()
            TIContinuousStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldPattern(
                    name="f0", typename="basic", bytes_expected=4
                ),
                f1=TIFieldPattern(
                    name="f1", typename="basic", bytes_expected=2
                ),
                f2=TIFieldPattern(
                    name="f2", typename="basic", bytes_expected=0
                ),
                f3=TIFieldPattern(
                    name="f3", typename="basic", bytes_expected=3
                ),
                f4=TIFieldPattern(
                    name="f4", typename="basic", bytes_expected=4
                ),
            )._modify_additions(adds)
            compare_subs(dict(
                f0=dict(name="f0", start=0),
                f1=dict(name="f1", start=4),
                f2=dict(name="f2", start=6, stop=-7),
                f3=dict(name="f3", start=-7),
                f4=dict(name="f4", start=-4),
            ), adds)

        with self.subTest(test="last dyn"):
            adds = Additions(lower={"f2": Additions(current={"req": 1})})
            TIContinuousStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldPattern(
                    name="f0", typename="basic", bytes_expected=4
                ),
                f1=TIFieldPattern(
                    name="f1", typename="basic", bytes_expected=2
                ),
                f2=TIFieldPattern(
                    name="f2", typename="basic", bytes_expected=0
                ),
            )._modify_additions(adds)
            compare_subs(dict(
                f0=dict(name="f0", start=0),
                f1=dict(name="f1", start=4),
                f2=dict(name="f2", start=6, stop=None, req=1),
            ), adds)

        with self.subTest(test="only dyn"):
            adds = Additions()
            TIContinuousStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldPattern(name="f0", typename="basic", bytes_expected=0),
            )._modify_additions(adds)
            compare_subs(dict(f0=dict(name="f0", start=0, stop=None)), adds)

        with self.subTest(test="without dyn"):
            adds = Additions()
            TIContinuousStructPattern(
                typename="basic", name="test"
            ).configure(
                f0=TIFieldPattern(name="f0", typename="basic", bytes_expected=4),
                f1=TIFieldPattern(name="f1", typename="basic", bytes_expected=2),
                f2=TIFieldPattern(name="f2", typename="basic", bytes_expected=3),
            )._modify_additions(adds)
            compare_subs(
                dict(
                    f0=dict(name="f0", start=0),
                    f1=dict(name="f1", start=4),
                    f2=dict(name="f2", start=6),
                ),
                adds,
            )

    def test__modify_sub_additions_exc(self) -> None:
        with self.subTest(test="two dynamic field"):
            with self.assertRaises(TypeError) as exc:
                TIContinuousStructPattern(
                    typename="basic", name="test",
                ).configure(
                    f0=TIFieldPattern(name="f0", typename="basic", bytes_expected=0),
                    f1=TIFieldPattern(name="f1", typename="basic", bytes_expected=0),
                )._modify_additions(Additions())
            self.assertEqual(
                "two dynamic field not allowed", exc.exception.args[0]
            )

import unittest

import numpy as np

from pyiak_instr_deprecation.core import Code
from pyiak_instr_deprecation.utilities import split_complex_dict, StringEncoder


class TestSplitComplexDict(unittest.TestCase):

    def test_basic_usage(self) -> None:
        self.assertDictEqual(
            {
                "a": {"a": 1},
                "b": {"b": {"b": {20: 10}}},
                "d": {"d": [1, 2, 3]}
            },
            split_complex_dict(dict(
                a__a=1, b__b__b={20: 10}, d__d=[1, 2, 3]
            ))
        )

    def test_without_sep(self) -> None:
        res, wo_sep = split_complex_dict(dict(a=20, b__b=20), without_sep="other")
        self.assertDictEqual({"b": {"b": 20}}, res)
        self.assertDictEqual({"a": 20}, wo_sep)

    def test_raises(self) -> None:
        with self.assertRaises(KeyError) as exc:
            split_complex_dict({"a": 20})
        self.assertEqual(
            "key 'a' does not have separator '__'",
            exc.exception.args[0]
        )

        with self.assertRaises(ValueError) as exc:
            split_complex_dict({"a": 20}, without_sep="test")
        self.assertEqual(
            "invalid attribute 'without_sep': "
            "'test' not in {'raise', 'other'}",
            exc.exception.args[0]
        )


class TestStringEncoder(unittest.TestCase):

    DATA = dict(
        empty_str=("", ""),
        letter=("a", "a"),
        word=("lol", "lol"),
        sentence=("lol kek!!!", "lol kek!!!"),
        bytes=("\\bts(1,3,255,50,120)", b"\x01\x03\xff\x32\x78"),
        int=("1", 1),
        int_neg=("-1", -1),
        int_str=("\\str(3)", "3"),
        float=("5.4321", 5.4321),
        float_neg=("-5.4321", -5.4321),
        float_str=("\\str(3.7)", "3.7"),
        efloat_small=("5.4321e-99", 5.4321e-99),
        efloat_small_neg=("-5.4321e-99", -5.4321e-99),
        efloat_str=("\\str(4.321e-107)", "4.321e-107"),
        efloat_large=("5.4321e+99", 5.4321e+99),
        true=("True", True),
        true_str=("\\str(True)", "True"),
        false=("False", False),
        false_str=("\\str(False)", "False"),
        none=("None", None),
        none_str=("\\str(None)", "None"),
        list_empty=("\\lst()", []),
        list_diff=(
            "\\lst(a,1,1.1,1.1e-99,1.1e+99)", ["a", 1, 1.1, 1.1e-99, 1.1e+99]
        ),
        list_str_int=("\\lst(\\str(1))", ["1"]),
        tuple_empty=("\\tpl()", ()),
        tuple_diff=(
            "\\tpl(a,1,1.1,1.1e-99,1.1e+99)", ("a", 1, 1.1, 1.1e-99, 1.1e+99)
        ),
        dict_empty=("\\dct()", {}),
        dict_diff=(
            "\\dct(a,1,1.1,1.1e-99,1.1e+99,1.1e+178)",
            {"a": 1, 1.1: 1.1e-99, 1.1e+99: np.float64(1.1e+178)}
        ),
        set_empty=("\\set()", set()),
        set_int=("\\set(1,2,3)", {1, 2, 3}),
        array_empty=("\\npa()", np.array([])),
        array_int=("\\npa(1,2)", np.array([1, 2])),
        array_float=("\\npa(1.1,2.2)", np.array([1.1, 2.2])),
        array_efloat=("\\npa(1.1e-99,2.2e+99)", np.array([1.1e-99, 2.2e+99]))
    )
    DATA_CHAIN = dict(
        ch1=("\\dct(a,\\dct(1,1.1))", {"a": {1: 1.1}}),
        ch2=("\\dct(\\str(0),\\lst(),2,\\set(1,2))", {"0": [], 2: {1, 2}}),
        ch3=(
            "\\lst(0,-1.1,\\npa(),\\npa(2),\\dct())",
            [0, -1.1, np.array([]), np.array([2]), {}]
        ),
        ch4=(
            "\\npa("
            "\\npa(\\npa(0,-1),\\npa(2,3)),"
            "\\npa(\\npa(4,5),\\npa(6,7))"
            ")",
            np.array([[[0, -1], [2, 3]], [[4, 5], [6, 7]]])
        )
    )

    def test_from_str_single(self) -> None:
        self.assertEqual(
            [1, 2, (3,)], StringEncoder.from_str(r"\lst(1,2,\tpl(3))")
        )

    def test_from_str(self):

        def check(result, true_value):
            if isinstance(result, np.ndarray):
                self.assertIsInstance(result, np.ndarray)
                self.assertTrue(np.isclose(true_value, result).all())
            else:
                self.assertEqual(true_value, result)

        for name, (src, true) in self.DATA.items():
            with self.subTest(name=name, true=true):
                check(StringEncoder.from_str(src), true)
        for name, (src, true) in self.DATA_CHAIN.items():
            with self.subTest(type="chain", name=name, true=true):
                if name in ("ch3",):
                    self.assertEqual(
                        str(StringEncoder.from_str(src)), str(true)
                    )
                else:
                    check(StringEncoder.from_str(src), true)

    def test_to_str_single(self) -> None:
        self.assertEqual(
            r"\lst(1,2,\tpl(3))", StringEncoder.to_str([1, 2, (3,)])
        )

    def test_to_str(self):

        for name, (true, src) in self.DATA.items():
            with self.subTest(name=name, true=true):
                self.assertEqual(
                    true, StringEncoder.to_str(src)
                )
        for name, (true, src) in self.DATA_CHAIN.items():
            with self.subTest(type="chain", name=name, true=true):
                self.assertEqual(
                    true, StringEncoder.to_str(src)
                )

    def test_to_str_invalid(self) -> None:
        self.assertEqual(
            "\\tpl\t1,2,3", StringEncoder.to_str("\\tpl\t1,2,3")
        )

    def test_to_str_code_support(self) -> None:
        self.assertEqual(
            "1", StringEncoder.to_str(Code.OK)
        )

    def test_decorate(self) -> None:
        self.assertEqual(
            "\\dct(1,2,3,4)",
            StringEncoder._decorate(Code.DICT, "1,2,3,4")
        )

    def test_find_border(self) -> None:
        self.assertTupleEqual(
            (4, 8),
            StringEncoder._find_border("\\tpl(1,2)")
        )
        self.assertTupleEqual(
            (4, 8),
            StringEncoder._find_border("\\lst(1,2),3,4)")
        )

    def test_iter_string(self) -> None:
        ref_values = ["1", "2", "\\tpl(3,4)", "gg"]
        for ref, res in zip(
            ref_values,
            StringEncoder._iter_string("1,2,\\tpl(3,4),gg")
        ):
            with self.subTest(ref=ref, res=res):
                self.assertEqual(ref, res)

    def test_read_header(self) -> None:
        self.assertTupleEqual(
            (Code.TUPLE, "1,2"),
            StringEncoder._read_header("\\tpl(1,2)")
        )

import unittest

import numpy as np

from pyinstr_iakoster.utilities import split_complex_dict, StringEncoder


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
        bytes=("\\bts\t1,3,255,50,120", b"\x01\x03\xff\x32\x78"),
        int=("1", 1),
        int_neg=("-1", -1),
        int_str=("\\str\t3", "3"),
        float=("5.4321", 5.4321),
        float_neg=("-5.4321", -5.4321),
        float_str=("\\str\t3.7", "3.7"),
        efloat_small=("5.4321e-99", 5.4321e-99),
        efloat_small_neg=("-5.4321e-99", -5.4321e-99),
        efloat_str=("\\str\t4.321e-107", "4.321e-107"),
        efloat_large=("5.4321e+99", 5.4321e+99),
        true=("True", True),
        true_str=("\\str\tTrue", "True"),
        false=("False", False),
        false_str=("\\str\tFalse", "False"),
        none=("None", None),
        none_str=("\\str\tNone", "None"),
        list_empty=("\\lst\t", []),
        list_diff=(
            "\\lst\ta,1,1.1,1.1e-99,1.1e+99", ["a", 1, 1.1, 1.1e-99, 1.1e+99]
        ),
        tuple_empty=("\\tpl\t", ()),
        tuple_diff=(
            "\\tpl\ta,1,1.1,1.1e-99,1.1e+99", ("a", 1, 1.1, 1.1e-99, 1.1e+99)
        ),
        dict_empty=("\\dct\t", {}),
        dict_diff=(
            "\\dct\ta,1,1.1,1.1e-99,1.1e+99,1.1e+178",
            {"a": 1, 1.1: 1.1e-99, 1.1e+99: np.float64(1.1e+178)}
        ),
        set_empty=("\\set\t", set()),
        set_int=("\\set\t1,2,3", {1, 2, 3}),
        array_empty=("\\npa\t", np.array([])),
        array_int=("\\npa\t1,2", np.array([1, 2])),
        array_float=("\\npa\t1.1,2.2", np.array([1.1, 2.2])),
        array_efloat=("\\npa\t1.1e-99,2.2e+99", np.array([1.1e-99, 2.2e+99]))
    )
    DATA_FROM = dict(
        from_list_with_pars=("\\lst len=5\t1,2,3,4,5", [1, 2, 3, 4, 5])
    )
    DATA_CHAIN = dict(
        ch1=("\\dct\ta,\\v(\\dct\t1,1.1)", {"a": {1: 1.1}}),
        ch2=("\\dct\t0,\\v(\\lst\t),2,\\v(\\set\t1,2)", {0: [], 2: {1, 2}}), # todo add str mark if '0' (or number)
        ch3=(
            "\\lst\t0,-1.1,\\v(\\npa\t),\\v(\\npa\t2),\\v(\\dct\t)",
            [0, -1.1, np.array([]), np.array([2]), {}]
        ),
        ch4=(
            "\\npa\t\\v(\\npa\t\\v(\\npa\t0,-1),\\v(\\npa\t2,3)),"
            "\\v(\\npa\t\\v(\\npa\t4,5),\\v(\\npa\t6,7))",
            np.array([[[0, -1], [2, 3]], [[4, 5], [6, 7]]])
        )
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
        for name, (src, true) in self.DATA_FROM.items():
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

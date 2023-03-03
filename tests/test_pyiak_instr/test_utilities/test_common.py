import unittest

from src.pyiak_instr.utilities import split_complex_dict


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
            ))[0]
        )

    def test_without_sep(self) -> None:
        res, wo_sep = split_complex_dict(
            dict(a=20, b__b=20), without_sep="other"
        )
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
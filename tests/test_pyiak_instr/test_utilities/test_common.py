import unittest

from src.pyiak_instr.exceptions import NotAmongTheOptions
from src.pyiak_instr.utilities import split_complex_dict


class TestSplitComplexDict(unittest.TestCase):

    def test_basic_usage(self) -> None:
        self.assertDictEqual(
            {
                "a": {"b": 1, "c": {"d": {20: 10}}},
                "b": {"c": [1, 2, 3]}
            },
            split_complex_dict(dict(
                a__b=1, a__c__d={20: 10}, b__c=[1, 2, 3]
            ))[0]
        )

    def test_without_sep(self) -> None:
        with self.subTest(without_sep="other"):
            res, wo_sep = split_complex_dict(
                dict(a=20, b__b=20), without_sep="other"
            )
            self.assertDictEqual({"b": {"b": 20}}, res)
            self.assertDictEqual({"a": 20}, wo_sep)

        with self.subTest(without_sep="ignore"):
            res, wo_sep = split_complex_dict(
                dict(a=20, b__b=20), without_sep="ignore"
            )
            self.assertDictEqual({"b": {"b": 20}}, res)
            self.assertDictEqual({}, wo_sep)

    def test_split_level(self) -> None:
        data = dict(b__b=1, c__c__c=2, d__d__d__d=3)
        res, _ = split_complex_dict(data, split_level=1)
        self.assertDictEqual(
            {"b": {"b": 1}, "c": {"c__c": 2}, "d": {"d__d__d": 3}}, res
        )

        res, _ = split_complex_dict(data, split_level=2)
        self.assertDictEqual(
            {"b": {"b": 1}, "c": {"c": {"c": 2}}, "d": {"d": {"d__d": 3}}}, res
        )

    def test_raises(self) -> None:
        with self.assertRaises(KeyError) as exc:
            split_complex_dict({"a": 20})
        self.assertEqual(
            "key 'a' does not have separator '__'",
            exc.exception.args[0]
        )

        with self.assertRaises(NotAmongTheOptions) as exc:
            split_complex_dict({"a": 20}, without_sep="test")
        self.assertIn(
            "'without_sep' option 'test' not in", exc.exception.args[0]
        )

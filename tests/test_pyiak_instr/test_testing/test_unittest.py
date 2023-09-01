import unittest
from typing import Any

import numpy as np
import pandas as pd

from src.pyiak_instr.testing._unittest import compare_values, validate_object, compare_objects


class TestInstance:

    def __init__(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)


class TestCompareValues(unittest.TestCase):

    def test_basic(self) -> None:
        for i, (ref, act) in enumerate((
            (np.array([0, 1, 2]), np.array([0, 1, 2])),
            (pd.DataFrame(data=[[2, 3], [2, 3]]), pd.DataFrame(data=[[2, 3], [2, 3]])),
            (pd.Series(data=[1]), pd.Series(data=[1])),
            (1, 1),
            ("lol", "lol"),
        )):
            compare_values(self, ref, act)


class TestValidateObject(unittest.TestCase):

    def test_basic(self) -> None:
        validate_object(
            self,
            TestInstance(a=32, _a=33, b=44),
            a=32,
            b=44,
        )

    def test_without_one(self) -> None:
        validate_object(
            self,
            TestInstance(a=32, _a=33, b=44),
            a=32,
            wo_attrs=["b"],
        )

    def test_without_check(self) -> None:
        validate_object(
            self,
            TestInstance(a=32, _a=33, b=44),
            a=32,
            all_attrs=False,
        )

    def test_not_all_exc(self) -> None:
        with self.assertRaises(AssertionError) as exc:
            validate_object(
                self,
                TestInstance(a=32, _a=33, b=44),
                a=32,
            )
        self.assertEqual(
            "the following parameters are not specified:\n{'b'}",
            exc.exception.args[0],
        )


class TestCompareObjects(unittest.TestCase):

    def test_basic(self) -> None:
        compare_objects(
            self,
            TestInstance(a=2, b=3),
            TestInstance(a=2, b=3),
        )

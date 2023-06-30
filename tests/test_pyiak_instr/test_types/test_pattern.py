import unittest
from typing import Any

import numpy as np

from ...utils import validate_object

from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.types import (
    Additions,
    Pattern,
    SurPattern,
)

from tests.pyiak_instr_ti.types.pattern import TIPattern, TISurPattern


class TestPattern(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            typename="basic",
            wo_attrs=["additions"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid typename"):
            with self.assertRaises(KeyError) as exc:
                TIPattern(typename="base")
            self.assertEqual("'base' not in {'basic'}", exc.exception.args[0])

        with self.subTest(test="without options"):
            with self.assertRaises(AttributeError) as exc:
                Pattern("basic")
            self.assertEqual(
                "'Pattern' object has no attribute '_options'",
                exc.exception.args[0],
            )

    def test_get(self) -> None:
        with self.subTest(test="basic"):
            self.assertDictEqual(
                {'a': 5, 'b': [], 'c': 11},
                self._instance().get(self._additions(c=11))
            )
        with self.subTest(test="with changes"):
            self.assertDictEqual(
                {'a': 11, 'b': [], 'c': 11},
                self._instance().get(self._additions(a=11, c=11))
            )

    def test_copy(self) -> None:
        exp = self._instance()
        act = exp.copy()

        self.assertIsNot(exp, act)
        self.assertDictEqual(exp.get(), act.get())

    def test_magic_init_kwargs(self) -> None:
        self.assertDictEqual(
            {"typename": "basic", "a": 5, "b": []},
            self._instance().__init_kwargs__()
        )

    def test_magic_getitem(self) -> None:
        self.assertEqual(5, self._instance()["a"])

    @staticmethod
    def _additions(**parameters: Any) -> Additions:
        return Additions(parameters)

    @staticmethod
    def _instance() -> TIPattern:
        return TIPattern("basic", a=5, b=[])


class TestEditableMixin(unittest.TestCase):

    def test_pop(self) -> None:
        obj = self._instance()
        self.assertListEqual([1, 2], obj.pop("b"))
        self.assertDictEqual(
            {"typename": "basic", "a": 5}, obj.__init_kwargs__()
        )

    def test_magic_setitem(self) -> None:
        obj = self._instance()
        obj["a"] = 3
        self.assertDictEqual({"a": 3, "b": [1, 2]}, obj._kw)

    @staticmethod
    def _instance() -> TIPattern:
        return TIPattern("basic", a=5, b=[1, 2])


class TestSurPattern(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            typename="basic",
            sub_pattern_names=[],
            wo_attrs=["additions"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(exc="_sub_p_type not exists"):
            with self.assertRaises(AttributeError) as exc:
                SurPattern("")
            self.assertEqual(
                "'SurPattern' object has no attribute '_sub_p_type'",
                exc.exception.args[0],
            )

    def test_configure(self) -> None:
        obj = self._instance()
        self.assertListEqual([], obj.sub_pattern_names)
        obj.configure(pattern=TIPattern("basic", a=0, req=""))
        self.assertListEqual(["pattern"], obj.sub_pattern_names)

    def test_get(self) -> None:
        self.assertDictEqual(
            dict(
                name="test",
                a=5,
                b=[],
                ii=99,
                subs={"f": {"a": 33, "i": 12, "req": ""}}
            ),
            self._instance().configure(
                f=TIPattern("basic", a=33, req=""),
            ).get(
                additions=Additions(
                    current={"ii": 99},
                    lower={"f": Additions(current={"i": 12})},
                )
            )
        )

    def test_get_exc(self) -> None:
        with self.subTest(exc="not configured"):
            with self.assertRaises(NotConfiguredYet) as exc:
                self._instance().get()
            self.assertEqual(
                "TISurPattern not configured yet", exc.exception.args[0]
            )

    def test_copy(self) -> None:
        exp = self._instance().configure(
            a=TIPattern(typename="basic", name="a", c=32)
        )
        act = exp.copy()

        self.assertIsNot(exp, act)
        self.assertIsNot(exp.sub_pattern("a"), act.sub_pattern("a"))
        self.assertDictEqual(exp.get(), act.get())

    def test_sub_pattern(self) -> None:
        self.assertDictEqual(
            dict(
                typename="basic",
                a=33,
                req="",
            ),
            self._instance().configure(
                f=TIPattern("basic", a=33, req=""),
            ).sub_pattern("f").__init_kwargs__()
        )

    def test_get_sub_pattern_type(self) -> None:
        self.assertIs(TIPattern, self._instance().sub_pattern_type())

    @staticmethod
    def _instance() -> TISurPattern:
        return TISurPattern(
            typename="basic", name="test", a=5, b=[]
        )

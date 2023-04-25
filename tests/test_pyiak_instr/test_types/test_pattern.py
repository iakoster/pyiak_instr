import unittest
from typing import Any

import numpy as np

from ...utils import validate_object

from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.types import (
    PatternABC,
    MetaPatternABC,
    EditablePatternABC,
)


class TIPatternABC(PatternABC[dict]):

    _options = {"basic": dict}

    _required_init_parameters = {"req"}


class TIEditablePatternABC(EditablePatternABC):

    def __init__(self):
        self._kw = {"a": 5, "b": [1, 2]}


class TIMetaPatternABC(MetaPatternABC[dict, TIPatternABC]):

    _options = {"basic": dict}
    _sub_p_type = TIPatternABC
    _sub_p_par_name = "_"


class TestPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance,
            typename="basic",
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid typename"):
            with self.assertRaises(KeyError) as exc:
                TIPatternABC(typename="base")
            self.assertEqual("'base' not in {'basic'}", exc.exception.args[0])

        with self.subTest(test="without required"):
            with self.assertRaises(TypeError) as exc:
                TIPatternABC(typename="basic")
            self.assertEqual(
                "{'req'} not represented in parameters", exc.exception.args[0]
            )

    def test_get(self) -> None:
        self.assertDictEqual(
            {'a': 5, 'b': [], 'c': 11, "req": "rep"},
            self._instance.get(c=11)
        )

    def test_get_changes_allowed(self) -> None:
        self.assertDictEqual(
            {'a': 11, 'b': [], 'c': 11, "req": "rep"},
            self._instance.get(True, a=11, c=11)
        )

    def test_get_exc(self) -> None:
        with self.subTest(test="repeat parameter"):
            with self.assertRaises(SyntaxError) as exc:
                self._instance.get(False, a=11)
            self.assertEqual(
                "keyword argument(s) repeated: a", exc.exception.args[0]
            )

    def test_magic_init_kwargs(self) -> None:
        self.assertDictEqual(
            {"typename": "basic", "a": 5, "b": [], "req": "rep"},
            self._instance.__init_kwargs__()
        )

    def test_magic_contains(self) -> None:
        obj = self._instance
        self.assertIn("a", obj)
        self.assertNotIn("c", obj)

    def test_magic_eq(self) -> None:
        self.assertEqual(self._instance, self._instance)
        self.assertNotEqual(
            self._instance, TIPatternABC(
                typename="basic", a=5, req="rep"
            )
        )
        self.assertNotEqual(self._instance, 1)

    def test_magic_getitem(self) -> None:
        self.assertEqual(5, self._instance["a"])

    def test_magic_eq_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            _ = TIPatternABC(
                typename="basic", a=np.array([1, 2]), req="rep"
            ) == TIPatternABC(
                typename="basic", a=np.array([1, 2]), req="rep"
            )
        self.assertEqual(
            "The truth value of an array with more than one element is "
            "ambiguous. Use a.any() or a.all()",
            exc.exception.args[0],
        )

    @property
    def _instance(self) -> TIPatternABC:
        return TIPatternABC(typename="basic", req="rep", a=5, b=[])


class TestEditablePatternABC(unittest.TestCase):

    def test_add(self) -> None:
        obj = self._instance
        obj.add("c", {1, 2})
        self.assertDictEqual(dict(a=5, b=[1, 2], c={1, 2}), obj._kw)

    def test_add_exc(self) -> None:
        with self.assertRaises(KeyError) as exc:
            self._instance.add("a", 3)
        self.assertEqual(
            "parameter 'a' in pattern already",
            exc.exception.args[0]
        )

    def test_pop(self) -> None:
        obj = self._instance
        self.assertListEqual([1, 2], obj.pop("b"))
        self.assertDictEqual({"a": 5}, obj._kw)

    def test_magic_setitem(self) -> None:
        obj = self._instance
        obj["a"] = 3
        self.assertDictEqual({"a": 3, "b": [1, 2]}, obj._kw)

    def test_magic_setitem_exc(self) -> None:
        with self.assertRaises(KeyError) as exc:
            self._instance["c"] = 0
        self.assertEqual("'c' not in parameters", exc.exception.args[0])

    @property
    def _instance(self) -> TIEditablePatternABC:
        return TIEditablePatternABC()


class TestMetaPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance,
            name="test",
            typename="basic",
        )

    def test_init_exc(self) -> None:
        with self.subTest(exc="_sub_p_par_name not exists"):
            with self.assertRaises(AttributeError) as exc:
                MetaPatternABC("", "")
            self.assertEqual(
                "'MetaPatternABC' object has no attribute '_sub_p_par_name'",
                exc.exception.args[0],
            )

    def test_configure(self) -> None:
        obj = self._instance
        pattern = TIPatternABC(typename="basic", a=0, req="")

        self.assertDictEqual({}, obj._sub_p)
        obj.configure(pattern=pattern)
        self.assertDictEqual({"pattern": pattern}, obj._sub_p)

    def test_get(self) -> None:
        self.assertDictEqual(
            dict(
                name="test",
                a=5,
                b=[],
                ii=99,
                _={"f": {"a": 33, "req": ""}}
            ),
            self._instance.configure(
                f=TIPatternABC("basic", a=33, req=""),
            ).get(ii=99)
        )

    def test_get_exc(self) -> None:
        with self.subTest(exc="not configured"):
            with self.assertRaises(NotConfiguredYet) as exc:
                self._instance.get()
            self.assertEqual(
                "TIMetaPatternABC not configured yet", exc.exception.args[0]
            )

    def test__modify_all(self) -> None:
        obj = self._instance.configure(
            f=TIPatternABC("basic", req=1), s=TIPatternABC("basic", req=1),
        )
        self.assertDictEqual(
            {"f": {"a": 1}, "s": {}},
            obj._modify_all(True, {"f": {"a": 1}})
        )

    @property
    def _instance(self) -> TIMetaPatternABC:
        return TIMetaPatternABC(
            typename="basic", name="test", a=5, b=[]
        )

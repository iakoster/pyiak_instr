import unittest
from typing import Any

import numpy as np

from ...utils import validate_object

from src.pyiak_instr.types import (
    PatternABC,
    MetaPatternABC,
    EditablePatternABC,
)


class PatternABCTestInstance(PatternABC[dict]):

    _options = {"base": dict}

    _only_auto_parameters = ("only_auto",)

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> dict:
        raise NotImplementedError()


class EditablePatternABCTestInstance(EditablePatternABC):

    def __init__(self):
        self._kw = {"a": 5, "b": [1, 2]}


class MetaPatternABCTestInstance(MetaPatternABC[dict, PatternABCTestInstance]):

    _options = {"base": dict}
    _sub_p_type = PatternABCTestInstance

    def get(
        self, changes_allowed: bool = False, **additions: Any
    ) -> dict:
        raise NotImplementedError()


class TestPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance,
            typename="base",
        )

    def test__get_parameters_dict(self) -> None:
        self.assertDictEqual(
            dict(a=5, b=[], c=11),
            self._instance._get_parameters_dict(False, {"c": 11}),
        )

    def test__get_parameters_dict_exc(self) -> None:
        with self.subTest(test="repeat parameter"):
            with self.assertRaises(SyntaxError) as exc:
                self._instance._get_parameters_dict(False, {"a": 11})
            self.assertEqual(
                "keyword argument(s) repeated: a", exc.exception.args[0]
            )

        with self.subTest(test="set auto parameter"):
            with self.assertRaises(TypeError) as exc:
                self._instance._get_parameters_dict(True, {"only_auto": 1})
            self.assertEqual(
                "'only_auto' can only be set automatically",
                exc.exception.args[0],
            )

    def test_magic_init_kwargs(self) -> None:
        self.assertDictEqual(
            {"typename": "base", "a": 5, "b": []},
            self._instance.__init_kwargs__()
        )

    def test_magic_contains(self) -> None:
        obj = self._instance
        self.assertIn("a", obj)
        self.assertNotIn("c", obj)

    def test_magic_eq(self) -> None:
        self.assertEqual(self._instance, self._instance)
        self.assertNotEqual(
            self._instance, PatternABCTestInstance(typename="base", a=5)
        )
        self.assertNotEqual(self._instance, 1)

    def test_magic_getitem(self) -> None:
        self.assertEqual(5, self._instance["a"])

    def test_magic_eq_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            _ = PatternABCTestInstance(
                typename="base", a=np.array([1, 2])
            ) == PatternABCTestInstance(typename="base", a=np.array([1, 2]))
        self.assertEqual(
            "The truth value of an array with more than one element is "
            "ambiguous. Use a.any() or a.all()",
            exc.exception.args[0],
        )

    @property
    def _instance(self) -> PatternABCTestInstance:
        return PatternABCTestInstance(typename="base", a=5, b=[])


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
    def _instance(self) -> EditablePatternABCTestInstance:
        return EditablePatternABCTestInstance()


class TestMetaPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance,
            name="test",
            typename="base",
        )

    def test_configure(self) -> None:
        obj = self._instance
        pattern = PatternABCTestInstance(typename="base", a=0)

        self.assertDictEqual({}, obj._sub_p)
        obj.configure(pattern=pattern)
        self.assertDictEqual({"pattern": pattern}, obj._sub_p)

    def test_magic_init_kwargs(self) -> None:
        self.assertDictEqual(
            {"typename": "base", "name": "test", "a": 5, "b": []},
            self._instance.__init_kwargs__()
        )

    @property
    def _instance(self) -> MetaPatternABCTestInstance:
        return MetaPatternABCTestInstance(
            typename="base", name="test", a=5, b=[]
        )

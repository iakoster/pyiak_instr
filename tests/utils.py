import inspect
from unittest import TestCase
from typing import Any

import numpy as np
import numpy.testing
import pandas as pd
import pandas.testing


__all__ = [
    "compare_objects", "compare_values", "get_object_attrs", "validate_object"
]


def get_object_attrs(
        obj: object, wo_attrs: list[str] | None = None, wo_consts: bool = True
) -> list[str]:
    if wo_attrs is None:
        wo_attrs = []

    def is_not_callable(f: Any) -> bool:
        return not (inspect.ismethod(f) or inspect.isfunction(f))

    def is_correct_name(name: str) -> bool:
        return not (name.startswith("_") or wo_consts and name.isupper())

    def is_attribute(name: str, f: Any) -> bool:
        return is_not_callable(f) and is_correct_name(name)

    props = [n for n, m in inspect.getmembers(obj) if is_attribute(n, m)]
    for attr in wo_attrs:
        if attr in props:
            props.pop(props.index(attr))
    return props


def compare_values(case: TestCase, ref: Any, res: Any) -> None:
    if isinstance(ref, np.ndarray):
        numpy.testing.assert_allclose(ref, res)
    elif isinstance(ref, pd.DataFrame):
        pandas.testing.assert_frame_equal(ref, res)
    elif isinstance(ref, pd.Series):
        pandas.testing.assert_series_equal(ref, res)
    elif not hasattr(ref, "__eq__"):
        if len(get_object_attrs(ref)):
            compare_objects(case, ref, res)
        else:
            raise ValueError("values cannot be compared")
    else:
        case.assertEqual(ref, res)


def validate_object(
        case: TestCase,
        obj: object,
        check_attrs: bool = True,
        wo_attrs: list[str] | None = None,
        wo_consts: bool = True,
        **attrs: Any,
):
    if check_attrs:
        case.assertListEqual(
            get_object_attrs(
                obj, wo_attrs=wo_attrs, wo_consts=wo_consts,
            ),
            [*attrs],
            "attributes list is differ to reference",
        )

    for attr, ref in attrs.items():
        with case.subTest(class_=obj.__class__.__name__, attr=attr):
            case.assertEqual(ref, getattr(obj, attr))


def compare_objects(
        case: TestCase,
        ref: object,
        res: object,
        attrs: list[str] = None,
        wo_attrs: list[str] | None = None,
        wo_consts: bool = True,
):
    case.assertIsInstance(res, ref.__class__)
    if attrs is None:
        attrs = get_object_attrs(ref, wo_attrs=wo_attrs, wo_consts=wo_consts)

    for attr in attrs:
        with case.subTest(
            ref_class=ref.__class__.__name__,
            res_class=res.__class__.__name__,
            attr=attr,
        ):
            compare_values(case, getattr(ref, attr), getattr(res, attr))

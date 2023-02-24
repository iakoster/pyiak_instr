import inspect
from unittest import TestCase
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing

from pyiak_instr_deprecation.communication import (
    FieldType,
    FieldMessage,
)


def get_object_attrs(
        obj: object, wo_attrs: list[str] = None, wo_consts: bool = True
) -> list[str]:
    if wo_attrs is None:
        wo_attrs = []

    def is_callable(f) -> bool:
        return inspect.ismethod(f) or inspect.isfunction(f)

    props = []
    for m in inspect.getmembers(obj):
        if m[0].startswith("_") \
                or is_callable(m[1]) \
                or wo_consts and m[0].isupper():
            continue
        props.append(m[0])

    for attr in wo_attrs:
        if attr in props:
            props.pop(props.index(attr))
    return props


def compare_values(case: TestCase, ref: Any, res: Any) -> None:
    if isinstance(ref, np.ndarray):
        case.assertTrue(np.all(ref, res))
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
        check_attrs: bool = False,
        wo_attrs: list[str] | None = None,
        wo_consts: bool = True,
        **attrs: Any
):
    if check_attrs:
        case.assertSetEqual(
            set(get_object_attrs(
                obj, wo_attrs=wo_attrs, wo_consts=wo_consts,
            )),
            set(attrs),
            "attributes list is differ to reference"
        )

    for attr, ref in attrs.items():
        with case.subTest(class_=obj.__class__.__name__, attr=attr):
            case.assertEqual(ref, getattr(obj, attr))


def validate_fields(
        case: TestCase,
        message: FieldMessage,
        fields: list[FieldType],
        attrs: list[str] = None,
        wo_attrs: list[str] | None = None,
        wo_consts: bool = True,

) -> None:
    msg_fields = list(message)
    case.assertEqual(len(fields), len(msg_fields))
    for ref, res in zip(fields, msg_fields):
        compare_objects(
            case,
            ref,
            res,
            attrs=attrs,
            wo_attrs=wo_attrs,
            wo_consts=wo_consts,
        )


def compare_objects(
        case: TestCase,
        ref: object,
        res: object,
        attrs: list[str] = None,
        wo_attrs: list[str] | None = None,
        wo_consts: bool = True,
):
    case.assertIs(ref.__class__, res.__class__)
    if attrs is None:
        attrs = get_object_attrs(
            ref, wo_attrs=wo_attrs, wo_consts=wo_consts
        )

    for attr in attrs:
        with case.subTest(
            ref_class=ref.__class__.__name__,
            res_class=res.__class__.__name__,
            attr=attr
        ):
            compare_values(case, getattr(ref, attr), getattr(res, attr))


def compare_messages(
        case: TestCase, ref: FieldMessage, res: FieldMessage, wo_attrs: list[str] = None
) -> None:
    if wo_attrs is None:
        wo_attrs = [
            "get",
            "has",
            "address",
            "operation",
            "data_length",
            "data",
            "parent",
        ]
    attrs = get_object_attrs(ref, wo_attrs=wo_attrs)

    compare_objects(case, ref, res, attrs=attrs)
    for ref_field, res_field in zip(ref, res):
        compare_objects(case, ref_field, res_field, wo_attrs=["parent"])
    with case.subTest(test="content"):
        case.assertEqual(str(ref), str(res))

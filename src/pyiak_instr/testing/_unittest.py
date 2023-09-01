"""Private module of ``pyiak_instr.testing``."""
from typing import Any
from unittest import TestCase

import numpy as np
import numpy.testing
import pandas as pd
import pandas.testing

from ._utils import get_object_attrs


__all__ = ["compare_values", "validate_object", "compare_objects"]


def compare_values(case: TestCase, ref: Any, act: Any) -> None:
    """
    Compare two values.

    Parameters
    ----------
    case : TestCase
        unittest TestCase instance.
    ref : Any
        reference value.
    act : Any
        actual value.
    """
    if np.ndarray in (type(ref), type(act)):
        numpy.testing.assert_allclose(ref, act)
    elif isinstance(ref, pd.DataFrame):
        pandas.testing.assert_frame_equal(ref, act)
    elif isinstance(ref, pd.Series):
        pandas.testing.assert_series_equal(ref, act)
    elif not hasattr(ref, "__eq__"):
        if len(get_object_attrs(ref)):
            compare_objects(case, ref, act)
    else:
        case.assertEqual(ref, act)


def validate_object(
    case: TestCase,
    obj: type | object,
    all_attrs: bool = True,
    wo_attrs: list[str] = None,
    wo_consts: bool = True,
    **attrs: Any,
) -> None:
    """
    Validate object.

    Compare attributes with values from `attrs` by key.

    Parameters
    ----------
    case : TestCase
        unittest TestCase instance.
    obj : type | object
        object to validating.
    all_attrs : bool, default=True
        if True - check that `attrs` has all attribute names.
    wo_attrs : list[str], default=None
       list of attributes that do not need to be checked.
    wo_consts : bool, default=True
        excludes constants from checking.
    **attrs : Any
        attributes to comparing.

    Raises
    ------
    AssertionError
        if `all_attrs` is True and not all attributes specified in `attrs`.
    """
    if all_attrs:
        diff = set(
            get_object_attrs(
                obj,
                wo_attrs=wo_attrs,
                wo_consts=wo_consts,
            )
        ) - set(attrs)
        if len(diff) != 0:
            raise AssertionError(
                f"the following parameters are not specified:\n{diff}"
            )

    for attr, ref in attrs.items():
        with case.subTest(class_=obj.__class__.__name__, attr=attr):
            compare_values(case, ref, getattr(obj, attr))


def compare_objects(
    case: TestCase,
    ref: type | object,
    act: type | object,
    attrs: list[str] = None,
    wo_attrs: list[str] = None,
    wo_consts: bool = True,
) -> None:
    """
    Compare `ref` and `act` objects that their attributes are identical.

    Parameters
    ----------
    case : TestCase
        unittest TestCase instance.
    ref : type | object
        reference object.
    act : type | object
        actual object.
    attrs : list[str], default=None
        attributes to comparing.
    wo_attrs : list[str], default=None
       list of attributes that do not need to be checked.
    wo_consts : bool, default=True
        excludes constants from checking.
    """
    case.assertIsInstance(act, ref.__class__)
    if attrs is None:
        attrs = get_object_attrs(ref, wo_attrs=wo_attrs, wo_consts=wo_consts)

    for attr in attrs:
        with case.subTest(
            ref_class=ref.__class__.__name__,
            res_class=act.__class__.__name__,
            attr=attr,
        ):
            compare_values(case, getattr(ref, attr), getattr(act, attr))

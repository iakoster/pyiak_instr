import inspect
from unittest import TestCase
from typing import Any

import numpy as np
import pandas as pd

from pyinstr_iakoster.core import Code
from pyinstr_iakoster.communication import (
    FieldSetter,
    Message,
    RegisterMap,
    MessageErrorMark,
    MessageFormat,
)


STATIC_SETTERS = [
    dict(
        address=FieldSetter.address(fmt=">I"),
        data_length=FieldSetter.data_length(
            fmt=">I", units=FieldSetter.WORDS
        ),
        operation=FieldSetter.operation(fmt=">I", desc_dict={"w": 0, "r": 1}),
        data=FieldSetter.data(expected=-1, fmt=">I")
    ),
]

MF_MSG_ARGS = [
    dict(
        emark=MessageErrorMark(
            operation="neq",
            start_byte=12,
            stop_byte=16,
            value=b"\x00\x00\x00\x01"
        ),
        mf_name="n0",
        splitable=True,
        slice_length=1024,
    ),
    dict(mf_name="n1", splitable=False, slice_length=1024),
    dict(mf_name="n2", splitable=False, slice_length=1024),
]


def get_setters_n0() -> dict[str, FieldSetter]:
    return STATIC_SETTERS[0]


def get_setters_n1(data__fmt: str = "B") -> dict[str, FieldSetter]:
    return dict(
        preamble=FieldSetter.static(fmt=">H", default=0xaa55),
        operation=FieldSetter.operation(
            fmt=">B", desc_dict={"wp": 1, "rp": 2, "wn": 3, "rn": 4}
        ),
        response=FieldSetter.response(
            fmt=">B", codes={0: Code.OK}, default=0, default_code=Code.ERROR
        ),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=data__fmt),
        crc=FieldSetter.crc(fmt=">H")
    )


def get_setters_n2(data__fmt: str = "B") -> dict[str, FieldSetter]:
    return dict(
        preamble=FieldSetter.static(fmt=">H", default=0xaa55),
        operation=FieldSetter.operation(fmt=">B", desc_dict={"r": 1, "w": 2}),
        response=FieldSetter.response(
            fmt=">B",
            codes={0: Code.OK, 4: Code.WAIT},
            default=0,
            default_code=Code.ERROR
        ),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=data__fmt),
        crc=FieldSetter.crc(fmt=">H")
    )


def get_msg_n0() -> Message:
    return Message(**MF_MSG_ARGS[0]).configure(**STATIC_SETTERS[0])


def get_msg_n1(data__fmt: str = "B") -> Message:
    return Message(**MF_MSG_ARGS[1])\
        .configure(**get_setters_n1(data__fmt=data__fmt))


def get_msg_n2(data__fmt: str = "B") -> Message:
    return Message(**MF_MSG_ARGS[2])\
        .configure(**get_setters_n2(data__fmt=data__fmt))


def unpack_setter(setter: FieldSetter) -> dict[str, Any]:
    return dict(special=setter.special, kwargs=setter.kwargs)


def get_mf_reference(mf: MessageFormat):
    return dict(
        msg_args=mf.msg_args,
        setters={n: unpack_setter(s) for n, s in mf.setters.items()}
    )


def _get_mf(mf: MessageFormat, ref: bool):
    if ref:
        return mf, get_mf_reference(mf)
    return mf


def get_mf_n0(get_ref: bool = True):
    return _get_mf(MessageFormat(
        **MF_MSG_ARGS[0], **STATIC_SETTERS[0]
    ), get_ref)


def get_mf_n1(get_ref: bool = True):
    return _get_mf(MessageFormat(
        **MF_MSG_ARGS[1], **get_setters_n1(data__fmt=">f")
    ), get_ref)


def get_mf_n2(get_ref: bool = True):
    return _get_mf(MessageFormat(
        **MF_MSG_ARGS[2], **get_setters_n2(data__fmt=">f")
    ), get_ref)


def get_register_map_data() -> pd.DataFrame:

    def get_line(i: int, args: tuple):
        return [f"t{i}", f"t_{i}", *args, f"Short {i}. Long."]

    df_data = pd.DataFrame(columns=RegisterMap.EXPECTED_COLUMNS)
    for i_addr, reg_args in enumerate([
        ("n0", 1, "rw", 1, None),
        ("n0", 0x200, "rw", 1, ">H"),
        ("n0", 0x10, "ro", 20, None),
        ("n0", 0x100, "wo", 5, None),
        ("n0", 0x1000, "rw", 7, None),
        ("n1", 0x500, "ro", 4, ">f"),
        ("n1", 0xf000, "rw", 6, ">I"),
        ("n2", 0x10, "rw", 1, None),
        ("n2", 0x11, "rw", 1, ">I")
    ]):
        df_data.loc[i_addr] = get_line(i_addr, reg_args)
    return df_data


def get_object_attrs(
        obj: object, wo_parent: bool = True, wo_consts: bool = True
) -> list[str]:

    def is_callable(f) -> bool:
        return inspect.ismethod(f) or inspect.isfunction(f)

    props = []
    for m in inspect.getmembers(obj):
        if m[0].startswith("_") \
                or is_callable(m[1]) \
                or wo_consts and m[0].isupper():
            continue
        props.append(m[0])

    if wo_parent and "parent" in props:
        props.pop(props.index("parent"))
    return props


def compare_values(case: TestCase, ref: Any, res: Any) -> None:
    if isinstance(ref, np.ndarray):
        case.assertTrue(np.all(ref, res))
    elif isinstance(ref, (pd.Series, pd.DataFrame)):
        case.assertTrue(np.all(ref.values == res.values))
    else:
        case.assertEqual(ref, res)


def validate_object(
        case: TestCase,
        obj: object,
        check_attrs: bool = False,
        wo_parent: bool = True,
        wo_consts: bool = True,
        **attrs: Any
):
    if check_attrs:
        case.assertSetEqual(
            set(get_object_attrs(
                obj, wo_parent=wo_parent, wo_consts=wo_consts,
            )),
            set(attrs),
            "attributes list is differ to reference"
        )

    for attr, ref in attrs.items():
        with case.subTest(class_=obj.__class__.__name__, attr=attr):
            case.assertEqual(ref, getattr(obj, attr))


def compare_objects(
        case: TestCase,
        ref: object,
        res: object,
        attrs: list[str] = None,
        wo_parent: bool = True,
        wo_consts: bool = True,
):
    case.assertIs(ref.__class__, res.__class__)
    if attrs is None:
        attrs = get_object_attrs(
            ref, wo_parent=wo_parent, wo_consts=wo_consts
        )

    for attr in attrs:
        with case.subTest(
            ref_class=ref.__class__.__name__,
            res_class=res.__class__.__name__,
            attr=attr
        ):
            compare_values(case, getattr(ref, attr), getattr(res, attr))


def compare_messages(case: TestCase, ref: Message, res: Message) -> None:
    attrs = get_object_attrs(ref)
    for miss in ["address", "operation", "data_length", "data"]:
        if miss in attrs:
            attrs.pop(attrs.index(miss))

    compare_objects(case, ref, res, attrs=attrs)
    for ref_field, res_field in zip(ref, res):
        compare_objects(case, ref_field, res_field)
    with case.subTest(test="content"):
        case.assertEqual(ref.hex(), res.hex())

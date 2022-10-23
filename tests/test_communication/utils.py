import unittest
from typing import Any

import pandas as pd

from pyinstr_iakoster.communication import (
    FieldSetter,
    Message,
    MessageErrorMark,
    MessageFormat,
    Field,
    FieldType,
    CrcField,
    Register,
    DataLengthField,
    OperationField,
    ResponseField,
)


def get_asm_msg() -> Message:
    return Message(format_name="asm", splitable=True).configure(
        address=FieldSetter.address(fmt=">I"),
        data_length=FieldSetter.data_length(
            fmt=">I", units=FieldSetter.WORDS
        ),
        operation=FieldSetter.operation(fmt=">I", desc_dict={"w": 0, "r": 1}),
        data=FieldSetter.data(expected=-1, fmt=">I")
    )


def get_kpm_msg(data_fmt: str = ">b") -> Message:
    return Message(format_name="kpm").configure(
        preamble=FieldSetter.static(fmt=">H", default=0xaa55),
        operation=FieldSetter.operation(
            fmt=">B", desc_dict={"wp": 1, "rp": 2, "wn": 3, "rn": 4}
        ),
        response=FieldSetter.single(fmt=">B", default=0),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=data_fmt),
        crc=FieldSetter.crc(fmt=">H")
    )


def get_mf_dict(
        special=None,
        **kwargs
):
    return dict(
        special=special,
        kwargs=kwargs
    )


def get_mf_asm(reference: bool = True):

    mf = MessageFormat(
        emark=MessageErrorMark(
            operation="neq",
            start_byte=12,
            stop_byte=16,
            value=b"\x00\x00\x00\x01"
        ),
        format_name="asm",
        splitable=True,
        slice_length=1024,
        address=FieldSetter.address(fmt=">I"),
        data_length=FieldSetter.data_length(
            fmt=">I", units=FieldSetter.WORDS
        ),
        operation=FieldSetter.operation(
            fmt=">I", desc_dict={"w": 0, "r": 1}
        ),
        data=FieldSetter.data(expected=-1, fmt=">I")
    )

    if reference:
        return mf, dict(
            msg_args=dict(
                format_name="asm", splitable=True, slice_length=1024
            ),
            setters=dict(
                address=get_mf_dict(special=None, fmt=">I", info=None),
                data_length=get_mf_dict(
                    special=None, fmt=">I", units=0x11, info=None, additive=0
                ),
                operation=get_mf_dict(
                    special=None,
                    fmt=">I",
                    desc_dict={"w": 0, "r": 1},
                    info=None
                ),
                data=get_mf_dict(
                    special=None, expected=-1, fmt=">I", info=None
                )
            )
        )
    return mf


def get_mf_kpm(reference: bool = True):

    mf = MessageFormat(
        emark=MessageErrorMark(
            operation="neq", field_name="response", value=[0]
        ),
        format_name="kpm",
        splitable=False,
        slice_length=1024,
        preamble=FieldSetter.static(fmt=">H", default=0xaa55),
        operation=FieldSetter.operation(
            fmt=">B", desc_dict={
                "wp": 1, "rp": 2, "wn": 3, "rn": 4
            }
        ),
        response=FieldSetter.single(fmt=">B", default=0),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=">f"),
        crc=FieldSetter.crc(fmt=">H")
    )

    if reference:
        return mf, dict(
            msg_args=dict(
                format_name="kpm", splitable=False, slice_length=1024
            ),
            setters=dict(
                preamble=get_mf_dict(
                    special="static", fmt=">H", default=0xaa55, info=None
                ),
                operation=get_mf_dict(
                    special=None,
                    fmt=">B",
                    desc_dict={"wp": 1, "rp": 2, "wn": 3, "rn": 4},
                    info=None
                ),
                response=get_mf_dict(
                    special="single",
                    fmt=">B",
                    default=0,
                    info=None,
                    may_be_empty=False,
                ),
                address=get_mf_dict(special=None, fmt=">H", info=None),
                data_length=get_mf_dict(
                    special=None, fmt=">H", units=0x10, info=None, additive=0,
                ),
                data=get_mf_dict(
                    special=None, expected=-1, fmt=">f", info=None
                ),
                crc=get_mf_dict(
                    special="crc",
                    fmt=">H",
                    algorithm_name="crc16-CCITT/XMODEM",
                    info=None
                )
            )
        )
    return mf


def get_register_map_data() -> pd.DataFrame:
    df_data = pd.DataFrame(
        columns=[
            "external_name",
            "name",
            "format_name",
            "address",
            "length",
            "reg_type",
            "data_fmt",
            "description"
        ]
    )
    data = [
        ("asm", 1, 1, "rw", None),
        ("asm", 0x10, 20, "ro", None),
        ("asm", 0x100, 5, "wo", None),
        ("asm", 0x200, 1, "rw", ">H"),
        ("asm", 0x1000, 7, "rw", None),
        ("kpm", 0x500, 4, "ro", ">f"),
        ("kpm", 0xf000, 6, "rw", ">I")
    ]
    for i_addr, reg_args in enumerate(data):
        df_data.loc[len(df_data)] = [
            f"tst_{i_addr}",
            f"test_{i_addr}",
            *reg_args,
            f"test address {i_addr}. Other description."
        ]
    return df_data


def get_field_attributes(field) -> list[str]:  # todo: make it auto
    attrs = [
        "bytesize",
        "content",
        "stop_byte",
        "expected",
        "finite",
        "fmt",
        "info",
        "name",
        "format_name",
        "start_byte",
        "words_count",
        "default",
        "may_be_empty",
        "slice",
    ]
    if isinstance(field, CrcField):
        attrs += [
            "algorithm",
            "algorithm_name",
        ]
    elif isinstance(field, DataLengthField):
        attrs += [
            "units",
            "additive",
        ]
    elif isinstance(field, OperationField):
        attrs += [
            "base",
            "desc",
            "desc_dict",
            "desc_dict_r",
        ]
    elif isinstance(field, ResponseField):
        attrs += [
            "codes",
            "default_code",
        ]
    return attrs


def get_message_attributes() -> list[str]:
    return [
        "format_name",
        "splitable",
        "slice_length",
        "have_infinite",
        "src",
        "dst",
    ]


def validate_object(
        case: unittest.TestCase,
        obj: object,
        **attrs: Any
):
    class_name = obj.__class__.__name__
    for attr, val in attrs.items():
        with case.subTest(class_=class_name, attr=attr):
            case.assertEqual(val, getattr(obj, attr))


def validate_field(
        case: unittest.TestCase,
        field: Field,
        check_attrs: bool = False,
        **attributes: Any
):
    if check_attrs:
        case.assertSetEqual(set(get_field_attributes(field)), set(attributes))

    class_name = field.__class__.__name__
    validate_object(case, field, **attributes)
    with case.subTest(class_=class_name, name="parent"):
        case.assertIs(field.parent, None)


def compare_objects(
        case: unittest.TestCase,
        ref: object,
        res: object,
        attrs: list[str],
):
    for attr in attrs:
        with case.subTest(res_class=res.__class__.__name__, attr=attr):
            case.assertEqual(getattr(ref, attr), getattr(res, attr))


def compare_registers(
        case: unittest.TestCase,
        ref: Register,
        res: Register,
) -> None:
    compare_objects(
        case, ref, res, list(get_register_map_data().columns) + [
            "short_description"
        ]
    )


def compare_fields(
        case: unittest.TestCase,
        ref: FieldType,
        res: FieldType,
        parent: Message = None
) -> None:
    attrs = get_field_attributes(ref)
    case.assertIs(ref.__class__, res.__class__)

    compare_objects(case, ref, res, attrs)
    if parent is not None:
        with case.subTest(name=ref.name, test="parent"):
            case.assertIs(parent, res.parent)


def compare_messages(
        case: unittest.TestCase,
        ref: Message,
        res: Message
):
    for attr in get_message_attributes():
        case.assertEqual(getattr(ref, attr), getattr(res, attr))
    for ref_field, res_field in zip(ref, res):
        compare_fields(case, ref_field, res_field, parent=res)
    with case.subTest(test="content"):
        case.assertEqual(ref.hex(), res.hex())


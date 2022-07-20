import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.communication import (
    MessageFormat,
    FieldSetter,
    PackageFormat
)


DATA_TEST_PATH = DATA_TEST_DIR / "test.json"


def get_mf_asm(reference: bool = True):

    mf = MessageFormat(
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
                address=dict(special=None, kwargs=dict(fmt=">I", info=None)),
                data_length=dict(special=None, kwargs=dict(
                    fmt=">I", units=0x11, info=None, additive=0,
                )),
                operation=dict(special=None, kwargs=dict(
                    fmt=">I", desc_dict={"w": 0, "r": 1}, info=None
                )),
                data=dict(special=None, kwargs=dict(
                    expected=-1, fmt=">I", info=None
                ))
            )
        )
    return mf


def get_mf_kpm(reference: bool = True):

    mf = MessageFormat(
        format_name="kpm",
        splitable=False,
        slice_length=1024,
        preamble=FieldSetter.static(fmt=">H", content=0xaa55),
        operation=FieldSetter.operation(
            fmt=">B", desc_dict={
                "wp": 1, "rp": 2, "wn": 3, "rn": 4
            }
        ),
        response=FieldSetter.single(fmt=">B"),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=">f"),
        crc=FieldSetter.single(fmt=">H")
    )

    if reference:
        return mf, dict(
            msg_args=dict(
                format_name="kpm", splitable=False, slice_length=1024
            ),
            setters=dict(
                preamble=dict(special="static", kwargs=dict(
                    fmt=">H", content=0xaa55, info=None
                )),
                operation=dict(special=None, kwargs=dict(
                    fmt=">B",
                    desc_dict={"wp": 1, "rp": 2, "wn": 3, "rn": 4},
                    info=None
                )),
                response=dict(special="single", kwargs=dict(
                    fmt=">B", info=None, may_be_empty=False,
                )),
                address=dict(special=None, kwargs=dict(fmt=">H", info=None)),
                data_length=dict(special=None, kwargs=dict(
                    fmt=">H", units=0x10, info=None, additive=0,
                )),
                data=dict(special=None, kwargs=dict(
                    expected=-1, fmt=">f", info=None
                )),
                crc=dict(special="single", kwargs=dict(
                    fmt=">H", info=None, may_be_empty=False
                ))
            )
        )
    return mf


class TestMessageFormat(unittest.TestCase):

    def test_init(self):

        for mf, ref_data in (get_mf_asm(), get_mf_kpm()):
            format_name = mf.msg_args["format_name"]

            with self.subTest(format_name=format_name):
                self.assertDictEqual(ref_data["msg_args"], mf.msg_args)

            with self.subTest(format_name=format_name, setter="all"):
                self.assertEqual(len(ref_data["setters"]), len(mf.setters))
                for (ref_name, ref_setter), (name, setter) in zip(
                    ref_data["setters"].items(), mf.setters.items()
                ):
                    with self.subTest(format_name=format_name, setter=name):
                        self.assertEqual(ref_name, name)
                        self.assertEqual(
                            ref_setter["special"], setter.special
                        )
                        self.assertDictEqual(
                            ref_setter["kwargs"], setter.kwargs
                        )


class TestPackageFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.pf = PackageFormat(
            asm=get_mf_asm(False),
            kpm=get_mf_kpm(False)
        )

    def test_write_read(self):
        self.pf.write(DATA_TEST_PATH)
        pf = PackageFormat.read(DATA_TEST_PATH)
        for name, ref_mf in self.pf.formats.items():
            mf = pf[name]
            with self.subTest(name=name):
                self.assertEqual(ref_mf.msg_args, mf.msg_args)

            with self.subTest(name=name, setter="all"):
                self.assertEqual(len(ref_mf.setters), len(mf.setters))
                for (ref_set_name, ref_setter), (set_name, setter) in zip(
                    ref_mf.setters.items(), mf.setters.items()
                ):
                    with self.subTest(name=name, setter=name):
                        self.assertEqual(name, name)
                        self.assertEqual(ref_setter.special, setter.special)

                        self.assertDictEqual(
                            {k: v for k, v in ref_setter.kwargs.items()
                             if v is not None},
                            setter.kwargs
                        )

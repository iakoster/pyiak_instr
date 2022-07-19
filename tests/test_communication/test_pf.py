import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.exceptions import FilepathPatternError
from pyinstr_iakoster.communication import MessageFormat, FieldSetter


DATA_TEST_PATH = DATA_TEST_DIR / "test.json"


class TestPackageFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.pf = MessageFormat(
            format_name="def",
            splitable=False,
            slice_length=1024,
            preamble=FieldSetter.static(fmt=">H", content=0xaa55),
            response=FieldSetter.single(fmt=">B"),
            address=FieldSetter.address(fmt=">I"),
            data_length=FieldSetter.data_length(
                fmt=">I", units=FieldSetter.WORDS
            ),
            operation=FieldSetter.operation(
                fmt=">I", desc_dict={"w": 0, "r": 1}
            ),
            data=FieldSetter.data(expected=-1, fmt=">I")
        )

    def test_init(self):
        with self.subTest(test="cls_settings"):
            self.assertDictEqual(
                dict(format_name="def", splitable=False, slice_length=1024),
                self.pf.message
            )

        setters = dict(
            preamble=FieldSetter.static(fmt=">H", content=0xaa55),
            response=FieldSetter.single(fmt=">B"),
            address=FieldSetter.address(fmt=">I"),
            data_length=FieldSetter.data_length(
                fmt=">I", units=FieldSetter.WORDS
            ),
            operation=FieldSetter.operation(
                fmt=">I", desc_dict={"w": 0, "r": 1}
            ),
            data=FieldSetter.data(expected=-1, fmt=">I")
        )
        for name, setter in setters.items():
            with self.subTest(test="fields", name=name):
                self.assertEqual(
                    setter.special, self.pf.setters[name].special
                )
                self.assertDictEqual(
                    setter.kwargs, self.pf.setters[name].kwargs
                )

    def test_write_pf_error(self):
        with self.subTest(type="not database"):
            with self.assertRaises(FilepathPatternError) as exc:
                self.pf.write_pf(DATA_TEST_DIR / "test.ini")
            self.assertEqual(
                r"The path does not lead to '\\S+.json$' file",
                exc.exception.args[0]
            )

    def test_write_read_pf(self):
        self.pf.write_pf(DATA_TEST_PATH)
        pf = MessageFormat.read_pf(DATA_TEST_PATH, "def")
        with self.subTest(type="msg_sets"):
            self.assertDictEqual(self.pf.message, pf.message)
        for name, r_setter in pf.setters.items():
            with self.subTest(type="setters", name=name):
                self.assertIn(name, self.pf.setters)
                w_setter = self.pf.setters[name]
                self.assertEqual(w_setter.special, r_setter.special)
                for k, v in w_setter.kwargs.items():
                    if v is not None:
                        with self.subTest(type="kwargs", key=k):
                            self.assertEqual(v, r_setter.kwargs[k])

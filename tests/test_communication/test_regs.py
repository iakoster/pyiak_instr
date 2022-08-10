import unittest

import numpy as np

from .utils import get_register_map_data, compare_registers, validate_object

from pyinstr_iakoster.communication import (
    Register,
    RegisterMap
)


class TestRegister(unittest.TestCase):

    def test_init(self):
        validate_object(
            self,
            Register(
                "lol",
                "kek",
                "mf",
                0xfdec,
                123,
                reg_type="rw",
                description="short desc. Long desc."
            ),
            address=0xfdec,
            description="short desc. Long desc.",
            external_name="lol",
            format_name="mf",
            length=123,
            name="kek",
            short_description="short desc"
        )

    def test_invalid_type(self):
        with self.assertRaises(TypeError) as exc:
            Register("", "", "", 0, 0, "re")
        self.assertEqual(
            "invalid register type: 're'", exc.exception.args[0]
        )


class TestRegisterMap(unittest.TestCase):

    DATA = get_register_map_data()

    def setUp(self) -> None:
        self.rm = RegisterMap(self.DATA)

    def test_get(self):
        names = np.append(
            self.DATA["name"].values, self.DATA["external_name"].values
        )
        for name in names:
            compare_registers(
                self,
                Register.from_series(
                    self.DATA[
                        (self.DATA["external_name"] == name) |
                        (self.DATA["name"] == name)
                        ].iloc[0]),
                self.rm.get(name)
            )

    def test_getattr(self):
        compare_registers(
            self,
            Register(
                "tst_3",
                "test_3",
                "asm",
                0x200,
                1,
                "rw",
                ">H",
                "test address 3. Other description."
            ),
            self.rm.test_3
        )

    def test_getitem(self):
        compare_registers(
            self,
            Register(
                "tst_5",
                "test_5",
                "kpm",
                0x500,
                4,
                "ro",
                ">f",
                "test address 5. Other description."
            ),
            self.rm["test_5"]
        )

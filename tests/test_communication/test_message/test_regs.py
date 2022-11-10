import unittest

import numpy as np

from ..utils import get_register_map_data, compare_registers, validate_object

from pyinstr_iakoster.communication import (
    Register,
    RegisterMap
)


class TestRegister(unittest.TestCase):

    def setUp(self) -> None:
        self.reg = Register(
            "lol",
            "kek",
            "mf",
            0x100,
            0x800,
            reg_type="rw",
            description="short desc. Long desc."
        )

    def test_init(self):
        validate_object(
            self,
            self.reg,
            address=0x100,
            description="short desc. Long desc.",
            external_name="lol",
            format_name="mf",
            length=2048,
            name="kek",
            short_description="short desc"
        )

    def test_invalid_type(self):
        with self.assertRaises(TypeError) as exc:
            Register("", "", "", 0, 0, "re")
        self.assertEqual(
            "invalid register type: 're'", exc.exception.args[0]
        )

    def test_shift(self):
        reg = self.reg.shift(0)
        compare_registers(self, self.reg, reg)
        reg = self.reg.shift(128)
        validate_object(
            self,
            reg,
            address=384,
            description="short desc. Long desc.",
            external_name="lol",
            format_name="mf",
            length=1920,
            name="kek_shifted",
            short_description="short desc"
        )
        self.assertEqual(2304, reg.address + reg.length)
        reg = reg + 128
        validate_object(
            self, reg, address=512, length=1792, name="kek_shifted",
        )
        self.assertEqual(2304, reg.address + reg.length)
        reg += 128
        validate_object(self, reg, address=640, length=1664)
        self.assertEqual(2304, reg.address + reg.length)

    def test_shift_exc(self):

        def test(shift: int, exc_msg: str):
            with self.subTest(shift=shift, exc_msg=exc_msg):
                with self.assertRaises(ValueError) as exc:
                    self.reg.shift(shift)
                self.assertEqual(exc_msg, exc.exception.args[0])

        test(-100, "invalid shift: -100 not in [0, 2048)")
        test(2048, "invalid shift: 2048 not in [0, 2048)")
        test(2050, "invalid shift: 2050 not in [0, 2048)")


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

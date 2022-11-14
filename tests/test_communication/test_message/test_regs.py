import unittest

import numpy as np
import pandas as pd

from ..utils import (
    get_register_map_data,
    get_mf_asm,
    compare_registers,
    compare_messages,
    validate_object
)

from pyinstr_iakoster.communication import (
    Register,
    RegisterMap
)


class TestRegister(unittest.TestCase):

    MF = get_mf_asm(reference=False)
    REF_SERIES = pd.Series(
        index=RegisterMap.EXPECTED_COLUMNS,
        data=(
            "lol",
            "kek",
            "mf",
            0x100,
            "ro",
            0x800,
            None,
            "short desc. Long desc.",
        )
    )

    def setUp(self) -> None:
        self.reg = Register(
            "lol",
            "kek",
            "mf",
            0x100,
            0x800,
            register_type="rw",
            description="short desc. Long desc.",
            mf=self.MF
        )
        self.ro_reg = Register(
            "lol",
            "kek",
            "mf",
            0x100,
            0x800,
            register_type="ro",
            description="short desc. Long desc.",
        )
        self.wo_reg = Register(
            "lol",
            "kek",
            "mf",
            0x100,
            0x800,
            register_type="wo",
            description="short desc. Long desc.",
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

    def test_shift_and_add(self):
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

    def test_read(self) -> None:
        ref = self.MF.get().set(address=0x100, data_length=0x800, operation=1)
        res = self.reg.read()
        compare_messages(self, ref, res)

    def test_read_update(self) -> None:
        ref = self.MF.get(
            data={"fmt": ">f"}
        ).set(address=0x100, data_length=0x200, operation=1)
        res = self.reg.read(data_length=0x200, data__fmt=">f")
        compare_messages(self, ref, res)

    def test_write(self) -> None:
        ref = self.MF.get().set(
            address=0x100, operation=0, data=[3, 2, 1]
        )
        res = self.reg.write(data=[3, 2, 1])
        compare_messages(self, ref, res)

    def test_write_update(self) -> None:
        ref = self.MF.get(
            data={"fmt": ">f"}
        ).set(address=0x100, operation=0, data=[0.1])
        res = self.reg.write([0.1], data__fmt=">f")
        compare_messages(self, ref, res)

    def test_from_series(self) -> None:
        res = Register.from_series(self.REF_SERIES)
        compare_registers(self, self.ro_reg, res)

    def test_series(self) -> None:
        self.assertTrue(
            np.all(self.REF_SERIES.values == self.ro_reg.series.values),
            f"\n{self.REF_SERIES}\n{self.ro_reg.series}"
        )

    def test_properties(self) -> None:
        props = dict(
            short_description="short desc"
        )
        for prop, value in props.items():
            self.assertEqual(value, getattr(self.reg, prop))

    def test_common_exceptions(self) -> None:
        with self.assertRaises(ValueError) as exc:
            self.reg.read(address=0x200)
        self.assertEqual(
            "setting the 'address' is not allowed", exc.exception.args[0]
        )

        with self.assertRaises(AttributeError) as exc:
            self.ro_reg.read()
        self.assertEqual(
            "message format not specified", exc.exception.args[0]
        )

    def test_read_exceptions(self):
        with self.assertRaises(TypeError) as exc:
            self.wo_reg.read()
        self.assertEqual("write only register", exc.exception.args[0])

    def test_write_exceptions(self):
        with self.assertRaises(TypeError) as exc:
            self.ro_reg.write(data=1)
        self.assertEqual("read only register", exc.exception.args[0])


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

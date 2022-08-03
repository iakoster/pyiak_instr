import unittest

import numpy as np
import pandas as pd

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.communication import (
    Register,
    RegisterMap
)


def get_data() -> pd.DataFrame:
    df_data = pd.DataFrame(
        columns=[
            "extended_name",
            "name",
            "address",
            "length",
            "message_format_name",
            "description"
        ]
    )
    data = [
        (1, 1, "asm"),
        (0x10, 20, "asm"),
        (0x100, 5, "asm"),
        (0x200, 1, "asm"),
        (0x1000, 7, "asm"),
        (0x500, 4, "kpm"),
        (0xf000, 2, "kpm")
    ]
    for i_addr, (addr, dlen, fmt_name) in enumerate(data):
        df_data.loc[len(df_data)] = [
            f"tst_{i_addr}",
            f"test_{i_addr}",
            addr,
            dlen,
            fmt_name,
            f"test address {i_addr}. Other description."
        ]
    return df_data


def compare_registers(
        case: unittest.TestCase,
        reference: Register,
        result: Register,
) -> None:
    attrs = [
        "address",
        "length",
        "description",
        "extended_name",
        "message_format_name",
        "name",
        "short_description"
    ]
    for attr in attrs:
        with case.subTest(attr=attr):
            case.assertEqual(getattr(reference, attr), getattr(result, attr))


class TestRegister(unittest.TestCase):

    def test_init(self):

        reg = Register(
            "lol",
            "kek",
            0xfdec,
            "mf",
            "short desc. Long desc."
        )

        attrs = dict(
            address=0xfdec,
            description="short desc. Long desc.",
            extended_name="lol",
            message_format_name="mf",
            name="kek",
            short_description="short desc."
        )
        for attr, val in attrs.items():
            with self.subTest(attr=attrs):
                self.assertEqual(val, getattr(reg, attr))


class TestRegisterMap(unittest.TestCase):

    DATA = get_data()

    def setUp(self) -> None:
        self.rm = RegisterMap(self.DATA)

    def test_get(self):
        names = np.append(
            self.DATA["name"].values, self.DATA["extended_name"].values
        )
        for name in names:
            compare_registers(
                self,
                Register.from_series(
                    self.DATA[
                        (self.DATA["extended_name"] == name) |
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
                0x200,
                "asm",
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
                0x500,
                "kpm",
                "test address 5. Other description."
            ),
            self.rm["test_5"]
        )

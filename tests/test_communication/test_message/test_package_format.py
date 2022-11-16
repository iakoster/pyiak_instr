import unittest

import pandas.testing

from tests.env_vars import TEST_DATA_DIR
from ..utils import (
    get_msg_n0,
    get_msg_n1,
    get_mf_n0,
    get_mf_n1,
    get_register_map_data,
    compare_objects,
    compare_messages,
)

from pyinstr_iakoster.communication import (
    RegisterMap,
    Register,
    PackageFormat
)


DATA_JSON_PATH = TEST_DATA_DIR / "test.json"
DATA_DB_PATH = TEST_DATA_DIR / "test.db"


class TestPackageFormat(unittest.TestCase):

    REG_MAP_DATA = get_register_map_data()

    def setUp(self) -> None:
        self.pf = PackageFormat(
            register_map=RegisterMap(self.REG_MAP_DATA),
            n0=get_mf_n0(False),
            n1=get_mf_n1(False)
        )

    def test_write_read(self):
        self.pf.write(DATA_JSON_PATH, DATA_DB_PATH)
        pf = PackageFormat.read(DATA_JSON_PATH)\
            .read_register_map(DATA_DB_PATH)
        for name, ref_mf in self.pf.formats.items():
            mf = pf.get_format(name)
            with self.subTest(name=name):
                self.assertEqual(ref_mf.message, mf.message)

            with self.subTest(name):
                self.assertDictEqual(ref_mf.arf.kwargs, mf.arf.kwargs)

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

        with self.subTest(test="register_map"):
            pandas.testing.assert_frame_equal(
                pf.register_map.table,
                self.pf.register_map.table,
                check_names=False
            )

    def test_get_asm_basic(self):
        ref = get_msg_n1().set(
            address=0x01020304,
            data_length=2,
            operation="w",
            data=[34, 52]
        )
        res = self.pf.get("asm").extract(
            b"\x01\x02\x03\x04\x00\x00\x00\x02\x00\x00\x00\x00"
            b"\x00\x00\x00\x22\x00\x00\x00\x34"
        )
        compare_messages(self, ref, res)

    def test_get_kpm_basic(self):
        ref = get_msg_n1().set(
            operation="wp",
            response=0,
            address=0x33,
            data_length=2,
            data=[17, 32],
            crc=0xedbc
        )
        res = self.pf.get("n1", data={"fmt": "B"}).extract(
            b"\xaa\x55\x01\x00\x00\x33\x00\x02\x11\x20\xed\xbc"
        )
        compare_messages(self, ref, res)

    def test_get_register(self):
        res = self.pf.get_register("tst_4")
        compare_objects(
            self,
            Register(
                "tst_4",
                "test_4",
                "n0",
                0x1000,
                7,
                "rw",
                description="test address 4. Other description."
            ),
            res
        )
        compare_messages(
            self,
            get_msg_n0().set(
                address=0x1000,
                data_length=1,
                operation=0,
                data=10
            ),
            res.write([10])
        )

    def test_getattr(self):
        compare_messages(
            self,
            get_msg_n1(data__fmt=">H").set(
                address=0xf000,
                operation="rp",
                data_length=6,
                data=[3, 11, 32]
            ),
            self.pf.test_6.read(
                data=[3, 11, 32], update={"data": {"fmt": ">H"}}
            )
        )

    def test_write_with_update(self):
        self.assertEqual(
            ">f",
            self.pf.test_0.read(update={"data": {"fmt": ">f"}}).data.fmt
        )

    def test_read_wo(self):
        with self.assertRaises(TypeError) as exc:
            self.pf.test_2.read()
        self.assertEqual(
            "writing only", exc.exception.args[0]
        )

    def test_write_ro(self):
        with self.assertRaises(TypeError) as exc:
            self.pf.test_1.write()
        self.assertEqual(
            "reading only", exc.exception.args[0]
        )

    def test_invalid_data_length(self):
        with self.assertRaises(ValueError) as exc:
            self.pf.test_1.read(21)
        self.assertEqual(
            "invalid data length: 21 > 20", exc.exception.args[0]
        )
        with self.assertRaises(ValueError) as exc:
            self.pf.test_2.write([0] * 6)
        self.assertEqual(
            "invalid data length: 6 > 5", exc.exception.args[0]
        )

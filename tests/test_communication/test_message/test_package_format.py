import shutil
import unittest
from copy import deepcopy

import pandas.testing

from tests.env_vars import TEST_DATA_DIR
from ..data import (
    SETTERS,
    PF,
    get_message,
)
from ..utils import (
    compare_objects,
    compare_messages,
)

from pyiak_instr.communication import (
    Register,
    PackageFormat,
)

TEST_DIR = TEST_DATA_DIR / __name__.split(".")[-1]
CFG_PATH = TEST_DIR / "cfg.ini"
DB_PATH = TEST_DIR / "pf.db"


class TestPackageFormat(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_write_read(self):
        PF.write(formats=CFG_PATH, registers=DB_PATH)
        pf = PackageFormat.read(formats=CFG_PATH, registers=DB_PATH)

        with self.subTest(test="register map"):
            pandas.testing.assert_frame_equal(
                    pf.register_map.table, PF.register_map.table
                )

        for name, mf in PF.message_format_map.formats.items():
            with self.subTest(test="message format", name=name):
                compare_objects(self, PF.message_format_map.formats[name], mf)

    def test_get_n0_basic(self):
        ref = get_message(0).set(
            address=0x01020304,
            data_length=2,
            operation="w",
            data=[34, 52]
        )
        res = PF.get("n0").extract(
            b"\x01\x02\x03\x04\x00\x00\x00\x02\x00\x00\x00\x00"
            b"\x00\x00\x00\x22\x00\x00\x00\x34"
        )
        compare_messages(self, ref, res)

    def test_get_n1_basic(self):
        setters = deepcopy(SETTERS[1])
        setters["data"].kwargs["fmt"] = "B"
        ref = get_message(1).configure(**setters).set(
            operation="wp",
            response=0,
            address=0x33,
            data_length=2,
            data=[17, 32],
            crc=0xdfaf
        )
        res = PF.get("n1", data={"fmt": "B"}).extract(
            b"\xaa\x55\x01\x00\x00\x33\x00\x02\x11\x20\xdf\xaf"
        )
        compare_messages(self, ref, res)

    def test_get_register(self):
        res = PF.get_register("t_4")
        compare_objects(
            self,
            Register(
                "t4",
                "t_4",
                "n0",
                0x1000,
                1024,
                "rw",
                description="Short 4. Long.",
                mf=PF.get_format("n0"),
            ),
            res
        )
        compare_messages(
            self,
            get_message(0).set(
                address=0x1000,
                data_length=1,
                operation=0,
                data=10
            ),
            res.write([10])
        )

    def test_getitem(self):
        setters = deepcopy(SETTERS[1])
        setters["data"].kwargs["fmt"] = ">H"
        compare_messages(
            self,
            get_message(1).configure(**setters).set(
                address=0xf000,
                operation="rp",
                data_length=6,
                data=[3, 11, 32]
            ),
            PF["t_6"].read(data=[3, 11, 32], data__fmt=">H")
        )

    def test_read_get_set_difference(self) -> None:
        ...
        # fixme: not equal because
        #  .set call data_length.update(),
        #  but in .read data_length is set from the register length
        # compare_messages(
        #     self,
        #     self.pf["t6"].read().set(data=1.47),
        #     self.pf["t6"].read(data=1.47),
        # )

    def test_write_get_set_difference(self) -> None:
        compare_messages(
            self,
            PF["t6"].write().set(data=1.47),
            PF["t6"].write(data=1.47),
        )

    def test_write_with_update(self):
        self.assertEqual(">f", PF["t_0"].read(data__fmt=">f").data.fmt)

    def test_common_exc(self):

        with self.subTest(test="write only"):
            with self.assertRaises(TypeError) as exc:
                PF["t_3"].read()
            self.assertEqual(
                "write only register", exc.exception.args[0]
            )

        with self.subTest(test="read only"):
            with self.assertRaises(TypeError) as exc:
                PF["t_2"].write()
            self.assertEqual(
                "read only register", exc.exception.args[0]
            )

        with self.subTest(test="read invalid length"):
            with self.assertRaises(ValueError) as exc:
                PF["t_2"].read(21)
            self.assertEqual(
                "data length mote than register length: 21 > 20",
                exc.exception.args[0],
            )

        with self.subTest(test="write invalid length"):
            with self.assertRaises(ValueError) as exc:
                PF["t_3"].write([0] * 6)
            self.assertEqual(
                "data length mote than register length: 6 > 5",
                exc.exception.args[0]
            )

import shutil
import unittest

import pandas.testing

from tests.env_vars import TEST_DATA_DIR
from ..utils import (
    get_msg_n0,
    get_msg_n1,
    get_msg_n2,
    get_mf_n0,
    get_mf_n1,
    get_mf_n2,
    get_register_map_data,
    compare_objects,
    compare_messages,
)

from pyinstr_iakoster.communication import (
    Register,
    RegisterMap,
    MessageFormat,
    MessageFormatMap,
    PackageFormat,
)

TEST_DIR = TEST_DATA_DIR / __name__.split(".")[-1]
CFG_PATH = TEST_DIR / "cfg.ini"
DB_PATH = TEST_DIR / "pf.db"
CFG_DICT = dict(
        master=dict(
            formats="\\lst\tn0,n1,n2",
        ),
        n0__message=dict(
            arf="\\dct\toperand,!=,"
                "value,\\v(\\bts\t0,0,0,1),"
                "start,12,"
                "stop,16",
            mf_name="n0",
            splitable="True",
            slice_length="1024",
        ),
        n0__setters=dict(
            address="\\dct\tspecial,None,fmt,>I",
            data_length="\\dct\tspecial,None,fmt,>I,units,17,additive,0",
            operation="\\dct\tspecial,None,fmt,>I,"
                      "desc_dict,\\v(\\dct\tw,0,r,1)",
            data="\\dct\tspecial,None,expected,-1,fmt,>I",
        ),
        n1__message=dict(
            arf="\\dct\t",
            mf_name="n1",
            splitable="False",
            slice_length="1024",
        ),
        n1__setters=dict(
            preamble="\\dct\tspecial,static,fmt,>H,default,43605",
            operation="\\dct\tspecial,None,fmt,>B,"
                      "desc_dict,\\v(\\dct\twp,1,rp,2,wn,3,rn,4)",
            response="\\dct\tspecial,response,"
                     "fmt,>B,"
                     "codes,\\v(\\dct\t0,1280),"
                     "default,0,"
                     "default_code,1282",
            address="\\dct\tspecial,None,fmt,>H",
            data_length="\\dct\tspecial,None,fmt,>H,units,16,additive,0",
            data="\\dct\tspecial,None,expected,-1,fmt,>f",
            crc="\\dct\tspecial,crc,fmt,>H,algorithm_name,crc16-CCITT/XMODEM",
        ),
        n2__message=dict(
            arf="\\dct\t",
            mf_name="n2",
            splitable="False",
            slice_length="1024",
        ),
        n2__setters=dict(
            operation="\\dct\tspecial,None,fmt,>B,"
                      "desc_dict,\\v(\\dct\tr,1,w,2)",
            response="\\dct\tspecial,response,"
                     "fmt,>B,"
                     "codes,\\v(\\dct\t0,1280,4,1281),"
                     "default,0,"
                     "default_code,1282",
            address="\\dct\tspecial,None,fmt,>H",
            data_length="\\dct\tspecial,None,fmt,>H,units,16,additive,0",
            data="\\dct\tspecial,None,expected,-1,fmt,>f",
            crc="\\dct\tspecial,crc,fmt,>H,algorithm_name,crc16-CCITT/XMODEM",
        )
    )
REF_FORMATS = {
    "n0": get_mf_n0(get_ref=False),
    "n1": get_mf_n1(get_ref=False),
    "n2": get_mf_n2(get_ref=False),
}


class TestPackageFormat(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def setUp(self) -> None:
        self.pf = PackageFormat(
            registers=RegisterMap(get_register_map_data()),
            formats=MessageFormatMap(**REF_FORMATS)
        )

    def test_write_read(self):
        self.pf.write(formats=CFG_PATH, registers=DB_PATH)
        pf = PackageFormat.read(formats=CFG_PATH, registers=DB_PATH)

        with self.subTest(test="register map"):
            pandas.testing.assert_frame_equal(
                    pf.register_map.table, self.pf.register_map.table
                )

        for name, mf in self.pf.message_format_map.formats.items():
            with self.subTest(test="message format", name=name):
                compare_objects(self, self.pf.message_format_map.formats[name], mf)

    def test_get_n0_basic(self):
        ref = get_msg_n0().set(
            address=0x01020304,
            data_length=2,
            operation="w",
            data=[34, 52]
        )
        res = self.pf.get("n0").extract(
            b"\x01\x02\x03\x04\x00\x00\x00\x02\x00\x00\x00\x00"
            b"\x00\x00\x00\x22\x00\x00\x00\x34"
        )
        compare_messages(self, ref, res)

    def test_get_n1_basic(self):
        ref = get_msg_n1(data__fmt="B").set(
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
        res = self.pf.get_register("t_4")
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
                mf=self.pf.get_format("n0"),
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

    def test_getitem(self):
        compare_messages(
            self,
            get_msg_n1(data__fmt=">H").set(
                address=0xf000,
                operation="rp",
                data_length=6,
                data=[3, 11, 32]
            ),
            self.pf["t_6"].read(data=[3, 11, 32], data__fmt=">H")
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
            self.pf["t6"].write().set(data=1.47),
            self.pf["t6"].write(data=1.47),
        )

    def test_write_with_update(self):
        self.assertEqual(">f", self.pf["t_0"].read(data__fmt=">f").data.fmt)

    def test_common_exc(self):

        with self.subTest(test="write only"):
            with self.assertRaises(TypeError) as exc:
                self.pf["t_3"].read()
            self.assertEqual(
                "write only register", exc.exception.args[0]
            )

        with self.subTest(test="read only"):
            with self.assertRaises(TypeError) as exc:
                self.pf["t_2"].write()
            self.assertEqual(
                "read only register", exc.exception.args[0]
            )

        with self.subTest(test="read invalid length"):
            with self.assertRaises(ValueError) as exc:
                self.pf["t_2"].read(21)
            self.assertEqual(
                "data length mote than register length: 21 > 20",
                exc.exception.args[0],
            )

        with self.subTest(test="write invalid length"):
            with self.assertRaises(ValueError) as exc:
                self.pf["t_3"].write([0] * 6)
            self.assertEqual(
                "data length mote than register length: 6 > 5",
                exc.exception.args[0]
            )

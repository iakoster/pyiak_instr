import shutil
import unittest

from src.pyiak_instr.communication.message import (
    MessagePattern,
    MessageStructPattern,
    MessageFieldStructPattern,
)

from .....utils import validate_object
from ....env import TEST_DATA_DIR, get_local_test_data_dir
from .ti import TIPatternsMap


TEST_DIR = get_local_test_data_dir(__name__)


class TestPatternsMap(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_init(self) -> None:
        validate_object(
            self,
            TIPatternsMap(),
            pattern_names=[],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(KeyError) as exc:
            TIPatternsMap(
                MessagePattern.basic().configure(
                    s0=MessageStructPattern.basic()
                ),
                MessagePattern.basic().configure(
                    s0=MessageStructPattern.basic()
                ),
            )
        self.assertEqual(
            "pattern with name 's0' is already exists", exc.exception.args[0]
        )

    def test_write(self) -> None:
        self._instance().write(TEST_DIR / "test_write.ini")

        lines = [
            "[s0]",
            "_ = \dct(typename,basic)",
            "s0 = \dct(typename,basic,divisible,False,mtu,1500)",
            "f0 = \dct(typename,static,bytes_expected,1,fmt,\cod(520),"
            r"order,\cod(1280),default,\bts(0))",
            "f1 = \dct(typename,data,fmt,\cod(520),order,\cod(1280),"
            r"bytes_expected,0,default,\bts())",
            "",
            "[s1]",
            "_ = \dct(typename,basic)",
            "s1 = \dct(typename,basic,divisible,False,mtu,1500)",
            "f0 = \dct(typename,id,bytes_expected,1,fmt,\cod(520),"
            r"order,\cod(1280),default,\bts())",
            "f1 = \dct(typename,data_length,bytes_expected,1,fmt,\cod(520),"
            "order,\cod(1280),behaviour,\cod(1536),units,\cod(257),"
            r"additive,0,default,\bts())",
            "f2 = \dct(typename,crc,bytes_expected,2,fmt,\cod(521),"
            r"order,\cod(1280),poly,4129,init,0,default,\bts(),"
            "wo_fields,\set())",
        ]

        with open(TEST_DIR / "test_write.ini", "r") as file:
            for i_line, act_line in enumerate(file.read().split("\n")):
                if len(act_line) == 0:
                    continue
                with self.subTest(line=i_line):
                    self.assertEqual(lines[i_line], act_line)

    def test_write_read(self) -> None:
        ref = self._instance()
        ref.write(TEST_DIR / "test_write_read.ini")
        res = ref.read(TEST_DIR / "test_write_read.ini")
        self.assertListEqual(ref.pattern_names, res.pattern_names)

    def test_magic_getitem(self) -> None:
        ref = self._instance()
        for name in ["s0", "s1"]:
            self.assertListEqual([name], ref[name].sub_pattern_names)

    @staticmethod
    def _instance() -> TIPatternsMap:
        return TIPatternsMap(
            MessagePattern.basic().configure(
                s0=MessageStructPattern.basic().configure(
                    f0=MessageFieldStructPattern.static(),
                    f1=MessageFieldStructPattern.data(),
                )
            ),
            MessagePattern.basic().configure(
                s1=MessageStructPattern.basic().configure(
                    f0=MessageFieldStructPattern.id_(),
                    f1=MessageFieldStructPattern.data_length(),
                    f2=MessageFieldStructPattern.crc(),
                )
            ),
        )

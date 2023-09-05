import unittest

from src.pyiak_instr.testing import validate_object
from ...env import get_local_test_data_dir, remove_test_data_dir

from tests.pyiak_instr_ti.communication.message import (
    TIMessagePattern,
    TIStructPattern,
    TIFieldPattern,
)
from tests.pyiak_instr_ti.communication.format import TIPatternMap


TEST_DIR = get_local_test_data_dir(__name__)


class TestPatternMap(unittest.TestCase):

    maxDiff = None

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_init(self) -> None:
        validate_object(
            self,
            TIPatternMap(),
            pattern_names=[],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(KeyError) as exc:
            TIPatternMap(
                TIMessagePattern.basic().configure(
                    s0=TIStructPattern.basic()
                ),
                TIMessagePattern.basic().configure(
                    s0=TIStructPattern.basic()
                ),
            )
        self.assertEqual(
            "pattern with name 's0' is already exists", exc.exception.args[0]
        )

    def test_get_pattern(self) -> None:
        ref = self._instance()
        for name in ["s0", "s1"]:
            self.assertListEqual(
                [name], ref.get_pattern(name).sub_pattern_names
            )

    def test_write(self) -> None:
        self._instance().write(TEST_DIR / "test_write.ini")

        lines = [
            "[s0]",
            "_ = \dct(typename,basic)",
            "s0 = \dct(typename,basic,divisible,False,mtu,1500)",
            "f0 = \dct(typename,static,direction,\cod(5),bytes_expected,1,"
            r"fmt,\cod(520),order,\cod(1280),default,\bts(0))",
            "f1 = \dct(typename,data,direction,\cod(5),fmt,\cod(520),"
            r"order,\cod(1280),bytes_expected,0,default,\bts())",
            "",
            "[s1]",
            "_ = \dct(typename,basic)",
            "s1 = \dct(typename,basic,divisible,False,mtu,1500)",
            "f0 = \dct(typename,id,direction,\cod(5),bytes_expected,1,"
            r"fmt,\cod(520),order,\cod(1280),default,\bts())",
            "f1 = \dct(typename,dynamic_length,direction,\cod(5),"
            "bytes_expected,1,fmt,\cod(520),order,\cod(1280),"
            "behaviour,\cod(1536),units,\cod(257),additive,0,"
            r"default,\bts(0))",
            "f2 = \dct(typename,crc,direction,\cod(5),bytes_expected,2,"
            "fmt,\cod(521),order,\cod(1280),poly,4129,init,0,"
            r"default,\bts(0),wo_fields,\set())",
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

    @staticmethod
    def _instance() -> TIPatternMap:
        return TIPatternMap(
            TIMessagePattern.basic().configure(
                s0=TIStructPattern.basic().configure(
                    f0=TIFieldPattern.static(),
                    f1=TIFieldPattern.data(),
                )
            ),
            TIMessagePattern.basic().configure(
                s1=TIStructPattern.basic().configure(
                    f0=TIFieldPattern.id_(),
                    f1=TIFieldPattern.dynamic_length(),
                    f2=TIFieldPattern.crc(),
                )
            ),
        )

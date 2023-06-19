import unittest

import pandas as pd

from src.pyiak_instr.core import Code

from ....utils import validate_object
from tests.pyiak_instr_ti.communication.message import (
    TIMessagePattern,
    TIStructPattern,
    TIFieldPattern,
)
from tests.pyiak_instr_ti.communication.format import (
    TIFormatMap,
    TIPatternMap,
    TIRegisterMap,
)


class TestFormatsMap(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            wo_attrs=["registers", "patterns"],
        )

    def test_parser_read(self) -> None:
        msg = self._instance()["test"].read()
        self.assertEqual(b"\x16\x00\x00", msg.content())
        for name, ref in dict(
            addr=b"\x16", oper=b"\x00", dlen=b"\x00", data=b""
        ).items():
            self.assertEqual(ref, msg.content(name))

    def test_parser_write(self) -> None:
        msg = self._instance()["test"].write([1, 2, 3, 255])
        self.assertEqual(b"\x16\x01\x04\x01\x02\x03\xff", msg.content())
        for name, ref in dict(
            addr=b"\x16", oper=b"\x01", dlen=b"\x04", data=b"\x01\x02\x03\xff"
        ).items():
            self.assertEqual(ref, msg.content(name))

    @staticmethod
    def _instance() -> TIFormatMap:
        return TIFormatMap(
            patterns=TIPatternMap(
                TIMessagePattern.basic().configure(
                    s1=TIStructPattern.basic().configure(
                        addr=TIFieldPattern.address(),
                        oper=TIFieldPattern.operation(),
                        dlen=TIFieldPattern.dynamic_length(),
                        data=TIFieldPattern.data(),
                    )
                ),
            ),
            registers=TIRegisterMap(
                table=pd.DataFrame(
                    columns=[
                        "pattern",
                        "name",
                        "address",
                        "length",
                        "rw_type",
                        "description",
                        "message",
                        "struct",
                        "fields",
                    ],
                    data=[
                        ["s1", "test", 22, 33, Code.ANY, "", "\dct()", "\dct()", "\dct()"],
                    ]
                )
            )
        )

import unittest

import pandas as pd

from src.pyiak_instr.core import Code
from src.pyiak_instr.communication.message import (
    MessagePattern,
    MessageStructPattern,
    MessageFieldStructPattern,
)

from .....utils import validate_object
from .ti import TIFormatsMap, TIPatternsMap, TIRegistersMap


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
    def _instance() -> TIFormatsMap:
        return TIFormatsMap(
            patterns=TIPatternsMap(
                MessagePattern.basic().configure(
                    s1=MessageStructPattern.basic().configure(
                        addr=MessageFieldStructPattern.address(),
                        oper=MessageFieldStructPattern.operation(),
                        dlen=MessageFieldStructPattern.dynamic_length(),
                        data=MessageFieldStructPattern.data(),
                    )
                ),
            ),
            registers=TIRegistersMap(
                table=pd.DataFrame(
                    columns=[
                        "pattern",
                        "name",
                        "address",
                        "length",
                        "rw_type",
                        "descriptions",
                    ],
                    data=[
                        ["s1", "test", 22, 33, Code.ANY, ""],
                    ]
                )
            )
        )

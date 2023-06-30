import unittest
from datetime import datetime, timedelta

from src.pyiak_instr.core import Code

from ....utils import validate_object
from ....pyiak_instr_ti.communication.connection import TIConnection
from ....pyiak_instr_ti.communication.message import (
    TIMessagePattern,
    TIStructPattern,
    TIFieldPattern,
)


class TestConnection(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIConnection(10),
            address=10,
            rx_timeout=timedelta(seconds=5),
            tx_timeout=timedelta(seconds=15),
            wo_attrs=["api"],
        )

    def test_receive(self) -> None:
        msg = self._pattern().get_for_direction(Code.RX)

        with TIConnection(
            12,
            "TIMessage(f0=10, f1=0, f2=3), src=10, dst=12",
            rx_msg=[
                (b"\x10\x00\x03", 10)
            ]
        ) as con:
            con.receive(msg)

    def test_transmit(self) -> None:
        msg = self._pattern().get_for_direction(Code.TX).encode(f2=[8, 15])
        msg.dst = 12

        with TIConnection(
            10,
            "TIMessage(f0=0, f2=8 F), src=10, dst=12",
        ) as con:
            con.transmit(msg)

    def test_transmit_receive(self) -> None:
        msg = self._pattern().get_for_direction(Code.TX).encode(f2=[8, 15])
        msg.dst = 10

        messages = [
            b"\x00\x00\x03",
        ]
        with TIConnection(
            12,
            "TIMessage(f0=0, f2=8 F), src=12, dst=10",
            "TIMessage(f0=0, f1=0, f2=3), src=10, dst=12",
            rx_msg=[
                (messages[0], 10),
            ]
        ) as con:
            for i, rx_ in enumerate(con.transimt_receive(msg)):
                with self.subTest(msg=i):
                    self.assertEqual(rx_.content(), messages[i])

    def test_transmit_receive_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            msg = self._pattern().get_for_direction(Code.TX)
            msg.src = 11
            TIConnection(10).transimt_receive(msg)
        self.assertEqual(
            "addresses in message and connection is not equal: 11 != 10",
            exc.exception.args[0]
        )

    def test_transmit_receive_joined(self) -> None:
        msg = self._pattern().get_for_direction(Code.TX).encode(b"\x10\x00\x01\x02\x03\x04\x05\x06")
        msg.dst = 10

        messages = [
            b"\x10\x00\x00\x01\x02",
            b"\x10\x00\x05\x06",
        ]
        with TIConnection(
            12,
            "TIMessage(f0=10, f2=0 1 2 3), src=12, dst=10",
            "TIMessage(f0=10, f1=0, f2=0 1 2), src=10, dst=12",
            "TIMessage(f0=10, f2=4 5 6), src=12, dst=10",
            "TIMessage(f0=10, f1=0, f2=5 6), src=10, dst=12",
            rx_msg=[
                (messages[0], 10),
                (messages[1], 10),
            ]
        ) as con:
            self.assertEqual(
                con.transmit_receive_joined(msg).content(),
                b"\x10\x00\x00\x01\x02\x05\x06",
            )

    def test_rules(self) -> None:
        msg = self._pattern().get_for_direction(Code.TX).encode(f2=[8, 15])
        msg.dst = 10

        with TIConnection(
            12,
            "TIMessage(f0=0, f2=8 F), src=12, dst=10",
            "TIMessage(f0=10, f1=1, f2=3), src=11, dst=12",
            "message received from 11, but expected from 10",
            "TIMessage(f0=10, f1=1, f2=3), src=10, dst=12",
            "answer with response: <Code.WAIT: 2>",
            "TIMessage(f0=10, f1=0, f2=3), src=10, dst=12",
            "different id (tx/rx): 0/16",
            "TIMessage(f0=0, f2=8 F), src=12, dst=10",
            "TIMessage(f0=0, f1=0, f2=2), src=10, dst=12",
            rx_msg=[
                (b"\x10\x01\x03", 11),
                (b"\x10\x01\x03", 10),
                (b"\x10\x00\x03", 10),
                None,
                (b"\x00\x00\x02", 10),
            ]
        ) as con:
            con.transimt_receive(msg)

    @staticmethod
    def _pattern() -> TIMessagePattern:
        return TIMessagePattern.basic().configure(
            s0=TIStructPattern.basic(
                divisible=True,
                mtu=5,
            ).configure(
                f0=TIFieldPattern.id_(typename="id", default=b"\x00"),
                f1=TIFieldPattern.response(
                    direction=Code.RX,
                    descs={0: Code.OK, 1: Code.WAIT, 2: Code.ERROR},
                ),
                f2=TIFieldPattern.data(),
            ),
        )

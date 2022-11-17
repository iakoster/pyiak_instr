import unittest
import datetime as dt
from typing import Any

import numpy as np

from pyinstr_iakoster.communication import (
    Message,
    Connection,
)

from ..utils import (
    PF,
    validate_object,
    compare_messages,
)


SRC_ADDRESS = ("127.0.0.1", 4242)
DST_ADDRESS = ("127.0.0.1", 4224)


class TestApi(object):

    def __eq__(self, other) -> bool:
        return isinstance(other, TestApi)


class ConnectionTestInstance(Connection):

    def __init__(
            self, case: unittest.TestCase, address: str = SRC_ADDRESS
    ):
        super().__init__(TestApi(), address)
        self._case = case
        self._i_tx, self._i_rx = 0, 0
        self._rx, self._tx = tuple(), tuple()
        self._rx_asym = []

    def close(self) -> None:
        pass

    def receive(self) -> tuple[bytes, Any]:
        msg = self._rx[self._i_rx].to_bytes()

        if len(self._rx_asym):
            if isinstance(self._rx_asym, list):
                i, asym = self._rx_asym[self._i_rx]
            else:
                i, asym = self._rx_asym
            msg = msg[:i] + asym + msg[i:]

        self._i_rx += 1
        return msg, DST_ADDRESS

    def setup(self, *args: Any, **kwargs: Any) -> "ConnectionTestInstance":
        return self

    def transmit(self, message: Message) -> None:
        compare_messages(self._case, self._tx[self._i_tx], message)
        self._i_tx += 1

    def _bind(self, address: Any) -> None:
        ...

    def set_rx_messages(
            self,
            *messages: Message,
            asymmetric: list[tuple[int, bytes]] | tuple[int, bytes] = None
    ) -> "ConnectionTestInstance":
        assert len(messages), "messages list is empty"
        if asymmetric is None:
            asymmetric = []

        self._rx = messages
        self._rx_asym = asymmetric
        self._drop_counters()
        return self

    def set_tx_messages(
            self, *messages: Message
    ) -> "ConnectionTestInstance":
        assert len(messages), "messages list is empty"
        self._tx = messages
        self._drop_counters()
        return self

    def _drop_counters(self) -> None:
        self._i_tx, self._i_rx = 0, 0

    def __enter__(self) -> "ConnectionTestInstance":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._i_tx != len(self._tx) or self._i_rx != len(self._rx):
            raise ValueError(
                "receive {}/{}, transmit {}/{}".format(
                    self._i_tx, len(self._tx), self._i_rx, len(self._rx)
                )
            )


class TestConnectionTestInstance(unittest.TestCase):

    def test_exit(self):
        with self.assertRaises(ValueError) as exc:
            with ConnectionTestInstance(self) as con:
                con.set_tx_messages(Message())
                con.set_rx_messages(Message())
        self.assertEqual("receive 0/1, transmit 0/1", exc.exception.args[0])


class TestConnection(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            ConnectionTestInstance(self),
            check_attrs=True,
            transmit_timeout=dt.timedelta(seconds=15),
            receive_timeout=dt.timedelta(seconds=5),
            address=('127.0.0.1', 4242),
            hapi=TestApi(),
        )

    def test_read_single(self) -> None:

        for data in (np.arange(1), np.arange(256), np.arange(255)):
            with self.subTest(data_length=len(data)):
                with ConnectionTestInstance(self) as con:
                    con.set_tx_messages(self.read("t4", len(data)))
                    con.set_rx_messages(
                        self.read("t4", len(data)).set(data=data),
                        asymmetric=(12, b"\x00\x00\x00\x01")
                    )

                    compare_messages(
                        self,
                        self.read("t4", len(data), ans=True).set(data=data),
                        self.send(con, self.read("t4", len(data)))
                    )

    def test_read_several_ints(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(
                self.read("t4", 256), self.read("t4", 256, shift=256)
            )
            con.set_rx_messages(
                self.read("t4", 256).set(data=range(256)),
                self.read("t4", 256, shift=256).set(data=range(256, 512)),
                asymmetric=(12, b"\x00\x00\x00\x01")
            )

            compare_messages(
                self,
                self.read("t4", 512, ans=True).set(data=range(512)),
                self.send(con, self.read("t4", 512))
            )

    def test_read_several_not_ints(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(
                self.read("t4", 256),
                self.read("t4", 256, shift=256),
                self.read("t4", 255, shift=512),
            )
            con.set_rx_messages(
                self.read("t4", 256).set(data=range(256)),
                self.read("t4", 256, shift=256).set(data=range(256, 512)),
                self.read("t4", 255, shift=512).set(data=range(512, 767)),
                asymmetric=(12, b"\x00\x00\x00\x01"),
            )

            compare_messages(
                self,
                self.read("t4", 767, ans=True).set(data=range(767)),
                self.send(con, self.read("t4", 767))
            )

    def test_read_no_cuttable(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(self.read("t6"))
            con.set_rx_messages(self.read("t6").set(data=1.4782))

            compare_messages(
                self,
                self.read("t6", ans=True).set(data=1.4782),
                self.send(con, self.read("t6"))
            )

    def test_write_single(self) -> None:

        for data in (np.arange(1), np.arange(256), np.arange(255)):
            with self.subTest(data_length=len(data)):
                with ConnectionTestInstance(self) as con:
                    con.set_tx_messages(self.write("t4", data))
                    con.set_rx_messages(
                        self.write("t4"),
                        asymmetric=(12, b"\x00\x00\x00\x01")
                    )

                    compare_messages(
                        self,
                        self.write("t4", ans=True, data_length=len(data)),
                        self.send(con, self.write("t4", data))
                    )

    @staticmethod
    def read(reg_name: str, *args, shift: int = 0, ans: bool = False, **kwargs) -> Message:
        if ans:
            src_dst = {"src": DST_ADDRESS, "dst": SRC_ADDRESS}
        else:
            src_dst = {"src": SRC_ADDRESS, "dst": DST_ADDRESS}
        return (PF[reg_name] + shift).read(
            *args, **kwargs
        ).set_src_dst(**src_dst)

    @staticmethod
    def send(con: ConnectionTestInstance, message: Message) -> Message:
        return con.send(message, PF.get_format(message.mf_name).arf)

    @staticmethod
    def write(reg_name: str, *args, shift: int = 0, ans: bool = False, **kwargs) -> Message:
        if ans:
            src_dst = {"src": DST_ADDRESS, "dst": SRC_ADDRESS}
        else:
            src_dst = {"src": SRC_ADDRESS, "dst": DST_ADDRESS}
        return (PF[reg_name] + shift).write(
            *args, **kwargs
        ).set_src_dst(**src_dst)

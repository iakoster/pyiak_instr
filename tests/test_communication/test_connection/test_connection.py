import time
import unittest
import datetime as dt
from typing import Any

import numpy as np

from pyiak_instr.communication import (
    FieldMessage,
    MessageType,
    Connection,
)

from ..data import (
    PF,
    get_message,
)
from ..utils import (
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
            self,
            case: unittest.TestCase,
            address: str = SRC_ADDRESS,
            receive_delay: float = 0,
            log_entries: list[str] = None
    ):
        super().__init__(TestApi(), address)
        self.set_timeouts(
            transmit_timeout=0.04,
            receive_timeout=0.02,
        )
        self._rx_delay = receive_delay
        self._case = case

        if log_entries is None:
            log_entries = []

        self._i_log, self._log_entries = 0, log_entries
        self._i_tx, self._i_rx = 0, 0
        self._rx, self._tx = tuple(), tuple()
        self._rx_asym = []

    def close(self) -> None:
        pass

    def receive(self) -> tuple[bytes, Any]:
        msg = self._rx[self._i_rx]
        if msg is None:
            self._i_rx += 1
            time.sleep(self._rx_delay)
            raise TimeoutError("timeout exceed")
        msg = msg.in_bytes()

        if len(self._rx_asym):
            if isinstance(self._rx_asym, list):
                i_asym = self._rx_asym[self._i_rx]
            else:
                i_asym = self._rx_asym

            if i_asym is not None:
                i, asym = i_asym
                msg = msg[:i] + asym + msg[i:]

        self._i_rx += 1
        time.sleep(self._rx_delay)
        return msg, DST_ADDRESS

    def setup(self, *args: Any, **kwargs: Any) -> "ConnectionTestInstance":
        return self

    def transmit(self, message: MessageType) -> None:
        compare_messages(self._case, self._tx[self._i_tx], message)
        self._i_tx += 1

    def _bind(self, address: Any) -> None:
        ...

    def _log_info(self, entry: str) -> None:
        if not len(self._log_entries):
            return

        if self._i_log >= len(self._log_entries):
            raise IndexError(f"there is no entry with index {self._i_log}")

        ref = self._log_entries[self._i_log]
        if ref is not None:
            assert entry == ref, f"entry {self._i_log}: {entry}"
        self._i_log += 1

    def set_rx_messages(
            self,
            *messages: MessageType | None,
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
            self, *messages: MessageType
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
                    self._i_rx, len(self._rx), self._i_tx, len(self._tx)
                )
            )


class TestConnectionTestInstance(unittest.TestCase):

    def test_exit(self):
        with self.assertRaises(ValueError) as exc:
            with ConnectionTestInstance(self) as con:
                con.set_tx_messages(FieldMessage())
                con.set_rx_messages(FieldMessage())
        self.assertEqual("receive 0/1, transmit 0/1", exc.exception.args[0])


class TestConnection(unittest.TestCase):

    maxDiff = None

    def test_init(self) -> None:
        validate_object(
            self,
            ConnectionTestInstance(self),
            check_attrs=True,
            transmit_timeout=dt.timedelta(milliseconds=40),
            receive_timeout=dt.timedelta(milliseconds=20),
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

    def test_read_no_cuttable(self) -> None:  # fixme: data_length is 6 when data is empty
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(self.read("t6"))
            con.set_rx_messages(self.read("t6").set(data=1))

            compare_messages(
                self,
                self.read("t6", ans=True).set(data=1),
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

    def test_write_several_ints(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(
                self.write("t4", range(256)),
                self.write("t4", range(256, 512), shift=256),
            )
            con.set_rx_messages(
                self.write("t4"),
                self.write("t4", shift=256),
                asymmetric=(12, b"\x00\x00\x00\x01")
            )

            compare_messages(
                self,
                self.write("t4", ans=True, data_length=512),
                self.send(con, self.write("t4", range(512)))
            )

    def test_write_several_not_ints(self):
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(
                self.write("t4", range(256)),
                self.write("t4", range(256, 511), shift=256),
            )
            con.set_rx_messages(
                self.write("t4", data_length=256),
                self.write("t4", shift=256, data_length=255),
                asymmetric=(12, b"\x00\x00\x00\x01")
            )

            compare_messages(
                self,
                self.write("t4", ans=True, data_length=511),
                self.send(con, self.write("t4", range(511)))
            )

    def test_write_no_cuttable(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(self.write("t8", 1.4782))
            con.set_rx_messages(self.write("t8", data_length=4))

            compare_messages(
                self,
                self.write("t8", ans=True, data_length=4),
                self.send(con, self.write("t8", 1.4782))
            )

    def test_asymmetric_error_with_first(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(self.write("t4", 0), self.write("t4", 0))
            con.set_rx_messages(
                self.write("t4", data_length=1),
                self.write("t4", data_length=1),
                asymmetric=[
                    (12, b"\x00\x00\x00\x00"),
                    (12, b"\x00\x00\x00\x01"),
                ]
            )

            compare_messages(
                self,
                self.write("t4", ans=True, data_length=1),
                self.send(con, self.write("t4", 0))
            )

    def test_asymmetric_all_error(self) -> None:
        with ConnectionTestInstance(self, receive_delay=0.02) as con:
            con.set_tx_messages(self.write("t4", 0), self.write("t4", 0))
            con.set_rx_messages(
                self.write("t4", data_length=1),
                self.write("t4", data_length=1),
                asymmetric=[
                    (12, b"\x00\x00\x00\x03"),
                    (12, b"\x00\x00\x00\x00"),
                ]
            )

            with self.assertRaises(ConnectionError) as exc:
                self.send(con, self.write("t4", 0))
            self.assertEqual(
                "time is up: received=2, invalid_received=2, transmitted=2",
                exc.exception.args[0]
            )

    def test_timeout_first(self) -> None:
        with ConnectionTestInstance(self, receive_delay=0.02) as con:
            con.set_tx_messages(self.write("t4", 0), self.write("t4", 0))
            con.set_rx_messages(
                None,
                self.write("t4", data_length=1),
                asymmetric=(12, b"\x00\x00\x00\x01"),
            )

            compare_messages(
                self,
                self.write("t4", ans=True, data_length=1),
                self.send(con, self.write("t4", 0))
            )

    def test_timeout_all(self) -> None:
        with ConnectionTestInstance(self, receive_delay=0.02) as con:
            con.set_tx_messages(self.write("t4", 0), self.write("t4", 0))
            con.set_rx_messages(None, None)

            with self.assertRaises(ConnectionError) as exc:
                self.send(con, self.write("t4", 0))
            self.assertEqual(
                "time is up: received=0, invalid_received=0, transmitted=2",
                exc.exception.args[0]
            )

    def test_skip_empty(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(self.write("t4", 0))
            con.set_rx_messages(
                get_message(0),
                self.write("t4"),
                asymmetric=[
                    None, (12, b"\x00\x00\x00\x01")
                ]
            )

            compare_messages(
                self,
                self.write("t4", ans=True, data_length=1),
                self.send(con, self.write("t4", 0))
            )

    def test_receive_from_not_expected(self) -> None:

        class ConnectionTestInstanceTemp(ConnectionTestInstance):

            def receive(self) -> tuple[bytes, Any]:
                time.sleep(self._rx_delay)
                return b"\xfa", SRC_ADDRESS

        with ConnectionTestInstanceTemp(
            self,
            receive_delay=0.02,
            log_entries=[
                "<StrongFieldMessage(address=1000, data_length=1, operation=0, data=0), "
                "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",

                "fa, src=('127.0.0.1', 4242), dst=('127.0.0.1', 4242)",

                "message received from ('127.0.0.1', 4242), "
                "but expected from ('127.0.0.1', 4224)",

                "<StrongFieldMessage(address=1000, data_length=1, operation=0, data=0), "
                "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",

                "fa, src=('127.0.0.1', 4242), dst=('127.0.0.1', 4242)",

                "message received from ('127.0.0.1', 4242), "
                "but expected from ('127.0.0.1', 4224)",
            ]
        ) as con:
            con.set_tx_messages(self.write("t4", 0), self.write("t4", 0))

            with self.assertRaises(ConnectionError) as exc:
                self.send(con, self.write("t4", 0))
            self.assertEqual(
                "time is up: received=2, invalid_received=2, transmitted=2",
                exc.exception.args[0]
            )

    def test_response_wait(self) -> None:
        with ConnectionTestInstance(self) as con:
            con.set_tx_messages(self.read("t7", 4))
            con.set_rx_messages(
                self.read("t7", 0).set(response=4),
                self.read("t7", 4, data=2.7),
            )

            compare_messages(
                self,
                self.read("t7", 4, ans=True, data=2.7),
                self.send(con, self.read("t7"))
            )

    def test_response_other(self) -> None:
        with ConnectionTestInstance(
            self,
            log_entries=[
                "<StrongFieldMessage(operation=1, response=0, address=10, "
                "data_length=4, data=EMPTY, crc=4647), "
                "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",

                "01 03 00 10 00 00 e8 11, "
                "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)",

                "<StrongFieldMessage(operation=1, response=3, address=10, "
                "data_length=0, data=EMPTY, crc=E811), "
                "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)>",

                "receive with code(s): <Code.UNDEFINED: 255>",

                "<StrongFieldMessage(operation=1, response=0, address=10, "
                "data_length=4, data=EMPTY, crc=4647), "
                "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",

                "01 00 00 10 00 04 40 2c cc cd 17 db, "
                "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)",

                "<StrongFieldMessage(operation=1, response=0, address=10, "
                "data_length=4, data=402CCCCD, crc=17DB), "
                "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)>",
            ]
        ) as con:
            con.set_tx_messages(self.read("t7", 4), self.read("t7", 4))
            con.set_rx_messages(
                self.read("t7", 0).set(response=3),
                self.read("t7", 4, data=2.7),
            )

            compare_messages(
                self,
                self.read("t7", 4, ans=True, data=2.7),
                self.send(con, self.read("t7"))
            )

    def test_several_responses(self) -> None:
        with ConnectionTestInstance(
            self,
            log_entries=[
                "<StrongFieldMessage(operation=1, response1=0, address=24, "
                "data_length=4, data=EMPTY, response2=0), "
                "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",

                None,

                "<StrongFieldMessage(operation=1, response1=3, address=24, "
                "data_length=0, data=EMPTY, response2=0), "
                "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)>",

                "receive with code(s): {'response1': <Code.ERROR: 3>, "
                "'response2': <Code.OK: 1>}",

                "<StrongFieldMessage(operation=1, response1=0, address=24, "
                "data_length=4, data=EMPTY, response2=0), "
                "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",

                None,

                "<StrongFieldMessage(operation=1, response1=0, address=24, "
                "data_length=4, data=2, response2=0), "
                "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)>",
            ]
        ) as con:
            con.set_tx_messages(self.read("t9", 4), self.read("t9", 4))
            con.set_rx_messages(
                self.read("t9", 0).set(response1=3),
                self.read("t9", 4, data=2.7),
            )

            compare_messages(
                self,
                self.read("t9", 4, ans=True, data=2.7),
                self.send(con, self.read("t9"))
            )

    def test_id_field_correct(self) -> None:
        with ConnectionTestInstance(
                self,
                log_entries=[
                    "<FieldMessage(id=12345678, address=1123, data=EMPTY), "
                    "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",
                    "12 34 56 78 00 00 11 23 00 00 01 4d, "
                    "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)",
                    "<FieldMessage(id=12345678, address=1123, data=14D), "
                    "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)>",
                ]
        ) as con:
            con.set_tx_messages(
                self.read("t10", id=0x12345678),
            )
            con.set_rx_messages(
                self.read("t10", id=0x12345678, data=333),
            )

            compare_messages(
                self,
                self.read("t10", ans=True, id=0x12345678, data=333),
                self.send(con, self.read("t10", id=0x12345678)),
            )

    def test_id_field_invalid(self) -> None:
        with ConnectionTestInstance(
                self,
                log_entries=[
                    "<FieldMessage(id=12345678, address=1123, data=EMPTY), "
                    "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",
                    "12 34 56 79 00 00 11 23 00 00 01 4d, "
                    "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)",
                    None,
                    "receive with code(s): <Code.INVALID_ID: 768>",
                    "12 34 56 78 00 00 11 23 00 00 01 4d, "
                    "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)",
                    None,
                ]
        ) as con:
            con.set_tx_messages(
                self.read("t10", id=0x12345678),
            )
            con.set_rx_messages(
                self.read("t10", id=0x12345679, data=333),
                self.read("t10", id=0x12345678, data=333),
            )
            self.send(con, self.read("t10", id=0x12345678))

    def test_field_message(self) -> None:
        with ConnectionTestInstance(
            self,
            log_entries=[
                "<FieldMessage(id=4D2, address=1123, data=EMPTY), "
                "src=('127.0.0.1', 4242), dst=('127.0.0.1', 4224)>",

                None,

                "<FieldMessage(id=4D2, address=1123, data=EMPTY), "
                "src=('127.0.0.1', 4224), dst=('127.0.0.1', 4242)>",
            ]
        ) as con:
            con.set_tx_messages(self.read("t10", 256, id=1234))
            con.set_rx_messages(self.read("t10", 256, id=1234))

            compare_messages(
                self,
                self.read("t10", 256, ans=True, id=1234),
                self.send(con, self.read("t10", id=1234))
            )

    def test_exc_logger_not_self(self) -> None:
        with self.assertRaises(ValueError) as exc:
            Connection(hapi=None, logger="self_")
        self.assertEqual(
            "invalid logger: self_ != 'self'", exc.exception.args[0]
        )

    def test_exc_without_address(self) -> None:
        with self.assertRaises(ConnectionError) as exc:
            Connection(None).send(FieldMessage())
        self.assertEqual(
            "address not specified", exc.exception.args[0]
        )

    def test_exc_invalid_message_src(self) -> None:
        with self.assertRaises(ConnectionError) as exc:
            ConnectionTestInstance(self).send(
                self.read("t1", ans=True)
            )
        self.assertEqual(
            "addresses in message and connection is not equal: "
            "('127.0.0.1', 4224) != ('127.0.0.1', 4242)",
            exc.exception.args[0]
        )

    @staticmethod
    def read(
            reg_name: str, *args, shift: int = 0, ans: bool = False, **kwargs
    ) -> MessageType:
        if ans:
            src_dst = {"src": DST_ADDRESS, "dst": SRC_ADDRESS}
        else:
            src_dst = {"src": SRC_ADDRESS, "dst": DST_ADDRESS}
        return (PF[reg_name] + shift).read(
            *args, **kwargs
        ).set_src_dst(**src_dst)

    @staticmethod
    def send(con: ConnectionTestInstance, message: MessageType) -> MessageType:
        return con.send(message, PF.get_format(message.mf_name).arf)

    @staticmethod
    def write(
            reg_name: str, *args, shift: int = 0, ans: bool = False, **kwargs
    ) -> MessageType:
        if ans:
            src_dst = {"src": DST_ADDRESS, "dst": SRC_ADDRESS}
        else:
            src_dst = {"src": SRC_ADDRESS, "dst": DST_ADDRESS}
        return (PF[reg_name] + shift).write(
            *args, **kwargs
        ).set_src_dst(**src_dst)

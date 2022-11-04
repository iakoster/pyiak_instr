import logging
import datetime as dt
from typing import Any

from .._msg import Message, MessageContentError
from .._pf import MessageErrorMark
from ...core import Code


__all__ = [
    "Connection"
]


class Connection(object):  # nodesc

    ADDRESS_TYPE = Message.ADDRESS_TYPE

    def __init__(
            self,
            hapi: Any,
            address: ADDRESS_TYPE | None = None,
            logger: logging.Logger | str | None = None
    ):
        if isinstance(logger, str) and logger != "self":
            raise ValueError("invalid logger: %s != 'self'" % logger)

        self._hapi = hapi
        self._logger = logging.getLogger(
            __name__
        ) if isinstance(logger, str) else logger

        if address is None:
            self._addr = None
        else:
            self.bind(address)

        self._tx_to = dt.timedelta(seconds=15)
        self._rx_to = dt.timedelta(seconds=5)

    def close(self) -> None:
        raise NotImplementedError()

    def setup(self, *args: Any, **kwargs: Any) -> "Connection":
        raise NotImplementedError()

    def _bind(self, address: ADDRESS_TYPE) -> None:
        raise NotImplementedError()

    def _receive(self) -> tuple[bytes, ADDRESS_TYPE]:
        raise NotImplementedError()

    def _transmit(self, message: Message) -> None:
        raise NotImplementedError()

    def bind(self, address: ADDRESS_TYPE) -> "Connection":
        self._addr = address
        self._bind(address)
        return self

    def send(
            self,
            message: Message,
            emark: MessageErrorMark = MessageErrorMark()
    ) -> Message:
        if self._addr is None:
            raise ConnectionError("no address specified")
        if message.src != self._addr:
            raise ConnectionError(
                "addresses in message and connection is not equal: "
                f"{message.src} != {self._addr}"
            )

        if message.operation.base == "w":
            answer = self._write(message, emark)
        elif message.operation.base == "r":
            answer = self._read(message, emark)
        else:
            raise MessageContentError(
                message.__class__.__name__,
                message.operation.name,
                clarification="unknown base"
            )
        return answer

    def set_timeouts(
            self,
            transmit_timeout: dt.datetime | int = 15,
            receive_timeout: dt.datetime | int = 5,
    ):
        if isinstance(transmit_timeout, int):
            transmit_timeout = dt.timedelta(seconds=transmit_timeout)
        if isinstance(receive_timeout, int):
            receive_timeout = dt.timedelta(seconds=receive_timeout)

        self._tx_to = transmit_timeout
        self._rx_to = receive_timeout

    def _log_info(self, entry: str) -> None:
        if self._logger is not None:
            self._logger.info(entry)

    def _read(self, msg: Message, emark: MessageErrorMark) -> Message:
        if not msg.splitable:
            return self._send(msg, emark)

        msg_gen = msg.split()
        answer = self._send(next(msg_gen), emark)
        for tx_msg in msg_gen:
            answer += self._send(tx_msg, emark)
        return answer

    def _send(self, msg: Message, emark: MessageErrorMark) -> Message:

        received = 0
        invalid_received = 0
        transmitted = 0
        transmit_start = dt.datetime.now()

        while transmit_start - dt.datetime.now() < self._tx_to:

            self._transmit(msg)
            transmitted += 1
            self._log_info(repr(msg))
            receive_start = dt.datetime.now()

            while receive_start - dt.datetime.now() < self._rx_to:

                try:
                    ans, rec_from = self._receive()
                    if len(ans) == 0:
                        continue
                    received += 1
                    self._log_info("{}, src={}, dst={}".format(
                        ans.hex(" "), rec_from, self._addr
                    ))

                except TimeoutError as exc:
                    self._log_info(repr(exc))

                else:
                    if msg.dst != rec_from:
                        invalid_received += 1
                        self._log_info(
                            f"message received from {rec_from}, "
                            f"but expected from {msg.dst}"
                        )
                        continue

                    ans, code = self._validate_bytes_message(msg, ans, emark)
                    if code == Code.ERROR:
                        continue

                    code = self._validate_message(ans, emark)
                    self._log_info(repr(ans))

                    if code == Code.OK:
                        return ans
                    elif code == Code.WAIT:
                        receive_start = dt.datetime.now()
                    else:
                        invalid_received += 1
                        self._log_info(f"receive with code(s): %r" % code)

        raise ConnectionError(
            "time is up: "
            "received={}, invalid_received={}, transmitted={}".format(
                received, invalid_received, transmitted
            )
        )

    def _validate_bytes_message(
            self,
            tx_msg: Message,
            rx_msg: bytes,
            emark: MessageErrorMark,
    ) -> tuple[Message, Code]:
        emark_exists = False
        if emark.bytes_required:
            rx_msg, emark_exists = emark.exists(rx_msg)
        # todo: check that bytes can be reformatted into a message
        rx_msg = tx_msg.get_same_instance().extract(rx_msg)\
            .set_addresses(src=tx_msg.dst, dst=self._addr)

        if emark_exists:
            return rx_msg, Code.ERROR
        return rx_msg, Code.OK

    def _validate_message(
            self, rx_msg: Message, emark: MessageErrorMark
    ) -> Code | list[Code]:

        codes = list(rx_msg.response_codes.values())
        if len(codes):
            if len(codes) == 1:
                return codes[0]

            ref_code = codes[0]
            for code in codes[1:]:
                if code != ref_code:
                    return codes
            return ref_code

        # else:
        #     return Code.OK  # todo: uncomment after fix emark

        emark_exists = False
        if not emark.bytes_required:
            _, emark_exists = emark.exists(rx_msg)  # todo: remove, response codes do it

        if emark_exists:
            return Code.ERROR
        return Code.OK

    def _write(self, msg: Message, emark: MessageErrorMark) -> Message:
        if not msg.splitable:
            return self._send(msg, emark)

        msg_gen = msg.split()
        answer = self._send(next(msg_gen), emark)
        for tx_msg in msg_gen:
            self._send(tx_msg, emark)
        return answer

    @property
    def address(self) -> ADDRESS_TYPE:
        return self._addr

    @property
    def hapi(self):
        return self._hapi

    @property
    def receive_timeout(self) -> dt.timedelta:
        return self._rx_to

    @property
    def transmit_timeout(self) -> dt.timedelta:
        return self._tx_to

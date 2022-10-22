import logging
from typing import Any

from .._msg import Message, MessageContentError
from .._pf import MessageErrorMark


class Connection(object):  # nodesc

    ADDRESS_TYPE = Any

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

    def send(self, message: Message, emark: MessageErrorMark = MessageErrorMark()) -> Message:
        if self._addr is None:
            raise ConnectionError("no address specified")
        if message.tx != self._addr:
            raise ConnectionError(
                "addresses in message and connection is not equal: "
                f"{message.tx} != {self._addr}"
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

    def _read(self, msg: Message, emark: MessageErrorMark) -> Message:
        if not msg.splitable:
            return self._send(msg, emark)

        msg_gen = msg.split()
        answer = self._send(next(msg_gen), emark)
        for tx_msg in msg_gen:
            answer += self._send(tx_msg, emark)
        return answer

    def _send(self, msg: Message, emark: MessageErrorMark) -> Message:
        ...

    def _write(self, msg: Message, emark: MessageErrorMark) -> Message:
        if not msg.splitable:
            return self._send(msg, emark)

        msg_gen = msg.split()
        answer = self._send(next(msg_gen), emark)
        for tx_msg in msg_gen:
            self._send(tx_msg, emark)
        return answer

    @property
    def address(self) -> Any:
        return self._addr

    @property
    def hapi(self):
        return self._hapi

from __future__ import annotations
import logging
import datetime as dt
from typing import Any, Generic

from .._message import (
    MessageType,
    AsymmetricResponseField,
)
from ...core import Code, UseApi, API_TYPE


# todo: check type annotation
__all__ = [
    "Connection"
]


class Connection(UseApi[API_TYPE], Generic[API_TYPE]):
    """
    Represents base class for messaging with devices via ethernet, UART etc.

    Parameters
    ----------
    api: Any
        High-level API (e.g. socket, Serial etc.).
    address: Any or None, default=None
        connection address. If specified call .bind method with address.
    logger: logging.Logger or str or None, default=None
        logger for logging transmitted and received messages. Log to
        transferred logger or create new logger if is equal to 'self'.
        Raises ValueError if transferred string that not equal to 'self'.
    """

    def __init__(
            self,
            api: Any,
            address: Any | None = None,
            logger: logging.Logger | str | None = None
    ):
        if isinstance(logger, str) and logger != "self":
            raise ValueError("invalid logger: %s != 'self'" % logger)

        self._api = api
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
        """
        Close the connection (api).
        """
        raise NotImplementedError()

    def receive(self) -> tuple[bytes, Any]:
        """
        Receive message by api.

        Returns
        -------
        tuple[bytes, Any]
            received message and the address where the message came from.
        """
        raise NotImplementedError()

    def setup(self, *args: Any, **kwargs: Any) -> Connection:
        """
        Setup api.

        Parameters
        ----------
        args: Any
            some arguments for setup.
        kwargs: Any
            some keyword arguments for setup.

        Returns
        -------
        Connection
            self instance.
        """
        raise NotImplementedError()

    def transmit(self, message: MessageType) -> None:
        """
        Transmit message by api.

        Information for transmit can be taken from the message
        (e.g. destination address).

        Parameters
        ----------
        message: MessageType
            transmission message.
        """
        raise NotImplementedError()

    def _bind(self, address: Any) -> None:
        """
        Bind address to api.

        Parameters
        ----------
        address: Any
            destination address.
        """
        raise NotImplementedError()

    def bind(self, address: Any) -> Connection:
        """
        Bind address.

        Also call ._bind method.

        Parameters
        ----------
        address: Any
            destination address.

        Returns
        -------
        Connection
            self instance.
        """
        self._addr = address
        self._bind(address)
        return self

    def send(
            self,
            message: MessageType,
            arf: AsymmetricResponseField = AsymmetricResponseField()
    ) -> MessageType:
        """
        Send message to destination address (see Message.dst).

        Parameters
        ----------
        message: MessageType
            transmitted message.
        arf: MessageErrorMark, default=MessageErrorMark()
            error mark for asymmetric response. Can be obtained from
            the current message format.

        Returns
        -------
        MessageType
            response message
        """
        if self._addr is None:
            raise ConnectionError("address not specified")
        if message.src != self._addr:
            raise ConnectionError(
                "addresses in message and connection is not equal: "
                f"{message.src} != {self._addr}"
            )

        answers = [self._send(msg, arf) for msg in message.split()]
        answer = answers[0]
        for part in answers[1:]:
            answer += part
        self._post_send_process(message, answer)

        return answer

    def set_timeouts(
            self,
            transmit_timeout: int | float | dt.timedelta = 15,
            receive_timeout: int | float | dt.timedelta = 5,
    ) -> None:
        """
        Set timeouts for waiting for a reply and sending a message.

        If the input value is of integer type, the timeout will be
        in seconds, and if it is of float type - in milliseconds.

        Parameters
        ----------
        transmit_timeout: int | float | datetime.timedelta, default=15
            the time for which the message transmission will be attempted
            until a response is received.
        receive_timeout: int | float | datetime.timedelta, default=15
            the time in which you expect to receive an answer.
        """

        def get_timedelta(value: dt.timedelta | int | float) -> dt.timedelta:
            if isinstance(value, int):
                value = dt.timedelta(seconds=value)
            elif isinstance(value, float):
                value = dt.timedelta(milliseconds=int(value * 1000))
            return value

        self._tx_to = get_timedelta(transmit_timeout)
        self._rx_to = get_timedelta(receive_timeout)

    def _log_info(self, entry: str) -> None:  # todo: tests for correct entries
        """
        Log entry to logger if logger is not None.

        Parameters
        ----------
        entry: str
            log entry.
        """
        if self._logger is not None:
            self._logger.info(entry)

    def _post_send_process(
            self, tx: MessageType, rx: MessageType
    ) -> None:
        """
        Process sum of all received messages.

        Parameters
        ----------
        tx: MessageType
            transmitted message.
        rx: MessageType
            received message.
        """
        if tx.has.OperationField \
                and tx.get.OperationField.base == "w":  # todo: base Code
            rx.get.DataLengthField.set(tx.get.DataLengthField.content)

    def _send(self, msg: MessageType, arf: AsymmetricResponseField) -> MessageType:
        """
        Send message and get response.

        Uses .transmit for transmit message and .receive for receive message.

        Log every received message. Ignore messages received not from
        the expected address.

        Parameters
        ----------
        msg: MessageType
            message for sending.
        arf: AsymmetricResponseField
            asymmetric error field.

        Returns
        -------
        MessageType
            response message.

        Raises
        ------
        ConnectionError
            if the timeout (transmit timeout) is reached.
        """

        received = 0
        invalid_received = 0
        transmitted = 0
        transmit_start = dt.datetime.now()

        while (dt.datetime.now() - transmit_start) < self._tx_to:

            self.transmit(msg)
            transmitted += 1
            self._log_info(repr(msg))
            transmit_again = False

            receive_start = dt.datetime.now()
            while not transmit_again \
                    and (dt.datetime.now() - receive_start) < self._rx_to:

                try:
                    ans, rec_from = self.receive()
                    if len(ans) == 0:
                        self._log_info(f"empty message from {rec_from}")
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

                    ans, code = self._validate_bytes_message(msg, ans, arf)
                    if code != Code.OK:
                        self._log_info("receive with code(s): %r" % code)
                        invalid_received += 1
                        transmit_again = True
                        continue

                    code = self._validate_message(ans)
                    self._log_info(repr(ans))

                    if code == Code.OK:
                        return ans
                    else:
                        self._log_info("receive with code(s): %r" % code)
                        if code == Code.WAIT:
                            receive_start = dt.datetime.now()
                        else:
                            invalid_received += 1
                            transmit_again = True

        raise ConnectionError(
            "time is up: "
            "received={}, invalid_received={}, transmitted={}".format(
                received, invalid_received, transmitted
            )
        )

    def _validate_bytes_message(
            self,
            tx: MessageType,
            rx: bytes,
            arf: AsymmetricResponseField,
    ) -> tuple[MessageType, Code]:
        """
        Validate raw received message.

        Code is OK if there is no errors and ERROR instead.

        Parameters
        ----------
        tx: MessageType
            transmitted message.
        rx: bytes
            raw received message.
        arf: AsymmetricResponseField
            asymmetric error mark.

        Returns
        -------
        tuple[Message, Code]
            message and code converted to a class (validation result).
        """
        is_error = False
        if not arf.is_empty:
            rx, is_error = arf.match(rx)
        rx = tx.get_instance().extract(rx) \
            .set_src_dst(src=tx.dst, dst=self._addr)

        if is_error:
            return rx, Code.ERROR
        return rx, Code.OK

    def _validate_message(self, rx: MessageType) -> Code | dict[str, Code]:
        """
        validate converted to a class message.

        Parameters
        ----------
        rx: MessageType
            received message.

        Returns
        -------
        Code | dict[str, Code]
            validation result in code. If message have ResponseField returns
            code from it. Returns all codes if there are several
            ResponseField and not all equal.
        """

        codes_dict = rx.response_codes
        codes = list(codes_dict.values())
        if len(codes):
            if len(codes) == 1:
                return codes[0]

            ref_code = codes[0]
            for code in codes[1:]:
                if code != ref_code:
                    return codes_dict
            return ref_code
        else:
            return Code.OK

    @property
    def address(self) -> Any:
        """
        Returns
        -------
        Any
            connection address.
        """
        return self._addr

    @property
    def api(self) -> Any:
        """
        Returns
        -------
        Any
            high-level API (e.g. socket)
        """
        return self._api

    @property
    def receive_timeout(self) -> dt.timedelta:
        """
        Returns
        -------
        datetime.timedelta
            receive timeout.
        """
        return self._rx_to

    @property
    def transmit_timeout(self) -> dt.timedelta:
        """
        Returns
        -------
        datetime.timedelta
            transmit timeout.
        """
        return self._tx_to

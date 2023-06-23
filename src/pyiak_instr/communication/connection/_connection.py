"""Private module of ``pyiak_instr.communication.connection``."""
import logging
import datetime as dt
from abc import abstractmethod
from typing import Any, Generic, Self, TypeVar

from ...core import Code
from ...typing import WithApi
from ..message import Message, Basic, Struct, MessagePattern


BasicT = TypeVar("BasicT", bound=Basic)
StructT = TypeVar("StructT", bound=Struct[Any])
PatternT = TypeVar("PatternT", bound=MessagePattern[Any, Any])
ApiT = TypeVar("ApiT")
AddressT = TypeVar("AddressT")


# todo: protocol
class EmptyLogger:
    """
    Empty logger instance.
    """

    def info(self, *args: Any, **kwargs: Any) -> None:
        """
        Empty function.

        Parameters
        ----------
        *args : Any
            function arguments.
        **kwargs : Any
            function kwargs.
        """


class Connection(WithApi[ApiT], Generic[ApiT, AddressT]):
    """
    Connection base class.

    Parameters
    ----------
    address : AddressT
        connection address.
    logger : Code | logging.Logger, default=Code.NONE
        logger instance or `Code` to generate logger.
    """

    _logger: logging.Logger | EmptyLogger

    def __init__(
        self,
        api: ApiT,
        address: AddressT,
        logger: Code | logging.Logger = Code.NONE,
    ) -> None:
        WithApi.__init__(self, api=api)

        if isinstance(logger, logging.Logger):
            self._logger = logger
        elif logger is Code.SELF:
            self._logger = logging.getLogger(__name__)
        else:
            self._logger = EmptyLogger()

        self._addr = address

        self._tx_to = dt.timedelta(seconds=15)
        self._rx_to = dt.timedelta(seconds=5)

    @abstractmethod
    def close(self) -> None:
        """Close the connection (api)"""

    @abstractmethod
    def direct_receive(self) -> tuple[bytes, AddressT]:
        """
        Receive message from Api.

        Returns
        -------
        tuple[bytes, AddressT]
            - received message;
            - source address of received message.

        Raises
        ------
        TimeoutError
            when the message waiting time expires.
        """

    @abstractmethod
    def direct_transmit(
        self, message: Message[BasicT, StructT, PatternT, AddressT]
    ) -> None:
        """
        Transmit `message` to Api.

        Parameters
        ----------
        message : Message[BasicT, StructT, PatternT, AddressT]
            message to transmit.
        """

    @abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> Self:
        """
        Set up the connection (api).

        Parameters
        ----------
        args : Any
            set up parameters.
        kwargs : Any
            set uo parameters.

        Returns
        -------
        Self
            self instance.
        """

    def receive(
        self, empty: Message[BasicT, StructT, PatternT, AddressT]
    ) -> None:
        """
        Receive message and fill `empty`.

        Parameters
        ----------
        empty : Message[BasicT, StructT, PatternT, AddressT]
            empty message for filling.
        """
        answer, address = self.direct_receive()
        empty.encode(answer)
        empty.src = address
        empty.dst = self._addr
        self._logger.info(str(empty))

    def transmit(
        self, message: Message[BasicT, StructT, PatternT, AddressT]
    ) -> None:
        """
        Transmit message and log it.

        Parameters
        ----------
        message : Message[BasicT, StructT, PatternT, AddressT]
            message to transmit.
        """
        self.direct_transmit(message)
        self._logger.info(str(message))

    def transimt_receive(
        self, message: Message[BasicT, StructT, PatternT, AddressT]
    ) -> list[Message[BasicT, StructT, PatternT, AddressT]]:
        """
        Send message and receive all answers (one to one tx message).

        Parameters
        ----------
        message : MessageT
            message to transmit.

        Returns
        -------
        MessageT
            joined received message.

        Raises
        ------
        ValueError
            if `message` source is not equal to connection address.
        """
        if message.src is None:
            message.src = self._addr

        elif message.src != self._addr:
            raise ValueError(
                "addresses in message and connection is not equal: "
                f"{message.src} != {self._addr}"
            )

        return [self._transmit_receive(msg) for msg in message.split()]

    def transmit_receive_joined(
        self, message: Message[BasicT, StructT, PatternT, AddressT]
    ) -> Message[BasicT, StructT, PatternT, AddressT]:
        """
        Send message and receive joined answer.

        Parameters
        ----------
        message : MessageT
            message to transmit.

        Returns
        -------
        MessageT
            joined received message.
        """
        return self.transimt_receive(message)[0]  # todo: method

    def set_timeouts(
        self,
        tx_timeout: int | float | dt.timedelta = 15,
        rx_timeout: int | float | dt.timedelta = 5,
    ) -> Self:
        """
        Set TX/RX timeouts is seconds (if int/float) or directly as timedelta.

        Parameters
        ----------
        tx_timeout : int | float | dt.timedelta, default=15
            timeout to transmit message.
        rx_timeout : int | float | dt.timedelta, default=5
            timeout to receive message.

        Returns
        -------
        Self
            self instance.
        """
        if not isinstance(tx_timeout, dt.timedelta):
            tx_timeout = dt.timedelta(seconds=tx_timeout)
        if not isinstance(rx_timeout, dt.timedelta):
            rx_timeout = dt.timedelta(seconds=rx_timeout)

        self._tx_to = tx_timeout
        self._rx_to = rx_timeout
        return self

    def _transmit_receive(
        self, msg: Message[BasicT, StructT, PatternT, AddressT]
    ) -> Message[BasicT, StructT, PatternT, AddressT]:
        """
        Transmit message and receive answer.

        Parameters
        ----------
        msg : Message[BasicT, StructT, PatternT, AddressT]
            message to transmit.

        Returns
        -------
        Message[BasicT, StructT, PatternT, AddressT]
            received message.

        Raises
        ------
        ConnectionError
            if the timeout (transmit timeout) is reached.
        """

        transmit_start = dt.datetime.now()

        while (dt.datetime.now() - transmit_start) < self._tx_to:
            self.transmit(msg)
            receive_start = dt.datetime.now()

            while (dt.datetime.now() - receive_start) < self._rx_to:
                try:
                    ans = msg.pattern.get_for_direction(Code.RX)
                    self.receive(ans)

                except TimeoutError:
                    break

                else:
                    if msg.dst != ans.src:
                        # pylint: disable=logging-fstring-interpolation
                        self._logger.info(
                            f"message received from {ans.src}, but expected "
                            f"from {msg.dst}"
                        )
                        continue

                    # validate message content
                    return ans  # type: ignore[no-any-return]

        raise ConnectionError(f"no answer received within {self._tx_to}")

    @property
    def address(self) -> AddressT:
        """
        Returns
        -------
        AddressT
            connection address.
        """
        return self._addr

    @property
    def rx_timeout(self) -> dt.timedelta:
        """
        Returns
        -------
        dt.timedelta
            timeout to receive message.
        """
        return self._rx_to

    @property
    def tx_timeout(self) -> dt.timedelta:
        """
        Returns
        -------
        dt.timedelta
            timeout to transmit message.
        """
        return self._tx_to

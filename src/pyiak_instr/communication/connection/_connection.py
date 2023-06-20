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
        super().__init__(api=api)

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
        return message

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

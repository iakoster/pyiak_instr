"""Private module of ``pyiak_instr.communication.connection``."""
import logging
from abc import abstractmethod
from typing import Any, Generic, TypeVar

from ...core import Code
from ...typing import WithApi
from ..message import Message, Basic, Struct, MessagePattern


BasicT = TypeVar("BasicT", bound=Basic)
StructT = TypeVar("StructT", bound=Struct[Any])
PatternT = TypeVar("PatternT", bound=MessagePattern[Any, Any])
AddressT = TypeVar("AddressT")
ApiT = TypeVar("ApiT")


class EmptyLogger(logging.Logger):
    """
    Empty logger instance.
    """

    def __init__(self) -> None:
        super().__init__("empty")


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

    def __init__(
        self, address: AddressT, logger: Code | logging.Logger = Code.NONE
    ) -> None:
        super().__init__(api=self._get_api())

        if isinstance(logger, logging.Logger):
            self._logger = logger
        elif logger is Code.SELF:
            self._logger = logging.getLogger(__name__)
        else:
            self._logger = EmptyLogger()

        self._addr = address

    @abstractmethod
    def _get_api(self) -> ApiT:
        """
        Get Api instance with parameters from class.

        Returns
        -------
        ApiT
            api instance.
        """

    def send(
        self, message: Message[BasicT, StructT, PatternT, AddressT]
    ) -> Message[BasicT, StructT, PatternT, AddressT]:
        """
        Send message and receive answer.

        Parameters
        ----------
        message : Message[BasicT, StructT, PatternT, AddressT]
            message to transmit.

        Returns
        -------
        Message[BasicT, StructT, PatternT, AddressT]
            joined received message.
        """
        return message

    @property
    def address(self) -> AddressT:
        """
        Returns
        -------
        AddressT
            connection address.
        """
        return self._addr

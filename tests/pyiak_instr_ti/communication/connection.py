from typing import Any, Self, Mapping
from logging import Logger

from src.pyiak_instr.communication.connection import Connection

from .message import TIMessage


class TILogger(Logger):

    def __init__(self, logs: tuple[str | None, ...]) -> None:
        super().__init__("test")

        self._logs = logs
        self._i = 0

    def info(
        self,
        msg: str,
        *args: object,
        exc_info: Any = ...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Mapping[str, object] | None = ...,
    ) -> None:
        ref = self._logs[self._i]
        self._i += 1
        if (msg % args) != ref:
            raise ValueError(
                f"invalid {self._i} message:\nexp:{ref}\nact:{msg}"
            )

    def __del__(self) -> None:
        if len(self._logs) != self._i:
            raise ValueError(
                f"not all logs showed: {self._i}/{len(self._logs)}"
            )


class TIApi:

    def __init__(self, address: int):
        self.address = address


class TIConnection(Connection[TIApi, int]):

    def __init__(
            self,
            address: int,
            *logs: str | None,
            rx_msg: list[tuple[bytes, int] | None] = None,
    ) -> None:
        super().__init__(
            api=TIApi(address),
            address=address,
            logger=TILogger(logs),
        )

        self._i = 0
        self._rx = [] if rx_msg is None else rx_msg

    def close(self) -> None:
        ...

    def direct_receive(self) -> tuple[bytes, int]:
        rec = self._rx[self._i]
        self._i += 1
        if rec is None:
            raise TimeoutError()
        return rec

    def direct_transmit(self, message: TIMessage) -> None:
        ...

    def setup(self, *args: Any, **kwargs: Any) -> Self:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

"""Private module of ``pyiak_instr.communication.format``."""
from typing import Any, Generic, TypeVar

from ..message import MessagePattern, Message
from ._pattern_map import PatternMap
from ._register_map import Register, RegisterMap


__all__ = ["FormatMap"]


MessageT = TypeVar("MessageT", bound=Message[Any, Any, Any, Any])
PatternMapT = TypeVar("PatternMapT", bound=PatternMap[Any])
RegisterMapT = TypeVar("RegisterMapT", bound=RegisterMap[Any])


class RegisterParser(Generic[MessageT]):
    """
    Represents parser to work with register.
    """

    def __init__(
        self,
        pattern: MessagePattern[Any, Any],
        register: Register[MessageT],
    ) -> None:
        self._pat = pattern
        self._reg = register

    def read(self, length: int = 0) -> MessageT:
        """
        Get message with READ operation.

        Parameters
        ----------
        length : int
            reading length.

        Returns
        -------
        MessageT
            message instance.
        """
        return self._reg.read(self._pat, dynamic_length=length)

    def write(self, data: Any) -> MessageT:
        """
        Get message with WRITE operation.

        Parameters
        ----------
        data : Any
            data to writing.

        Returns
        -------
        MessageT
            message instance.
        """
        return self._reg.write(self._pat, data)


class FormatMap(Generic[PatternMapT, RegisterMapT, MessageT]):
    """
    Represents class with patterns and registers container.
    """

    def __init__(
        self, patterns: PatternMapT, registers: RegisterMapT
    ) -> None:
        self._patterns = patterns
        self._registers = registers

    @property
    def patterns(self) -> PatternMapT:
        """
        Returns
        -------
        PatternMapT
            patterns container.
        """
        return self._patterns

    @property
    def registers(self) -> RegisterMapT:
        """
        Returns
        -------
        RegisterMapT
            registers container.
        """
        return self._registers

    def __getitem__(self, name: str) -> RegisterParser[MessageT]:
        """
        Get register parser with `name` register.

        Parameters
        ----------
        name : str
            name of register.

        Returns
        -------
        RegisterParser[MessageT]
            register parser.
        """
        register = self._registers.get_register(name)
        pattern = self._patterns.get_pattern(register.pattern)
        return RegisterParser(pattern, register)

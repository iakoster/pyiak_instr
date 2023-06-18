# """Private module of ``pyiak_instr.communication.format_map.types``."""
# from abc import ABC
# from typing import Any, Generic, TypeVar
#
# from ...message.types import MessagePattern, Message
# from ._patterns_map import PatternsMapABC
# from ._registers_map import RegisterABC, RegistersMapABC
#
#
# __all__ = ["FormatsMapABC"]
#
#
# MessageT = TypeVar("MessageT", bound=Message[Any, Any, Any, Any])
# PatternsMapT = TypeVar("PatternsMapT", bound=PatternsMapABC[Any])
# RegistersMapT = TypeVar("RegistersMapT", bound=RegistersMapABC[Any])
#
#
# class RegisterParser(ABC, Generic[MessageT]):
#     """
#     Represents parser to work with register.
#     """
#
#     def __init__(
#         self,
#         pattern: MessagePattern[Any, Any],
#         register: RegisterABC[MessageT],
#     ) -> None:
#         self._pat = pattern
#         self._reg = register
#
#     def read(self, length: int = 0) -> MessageT:
#         """
#         Get message with READ operation.
#
#         Parameters
#         ----------
#         length : int
#             reading length.
#
#         Returns
#         -------
#         MessageT
#             message instance.
#         """
#         return self._reg.read(self._pat, dynamic_length=length)
#
#     def write(self, data: Any) -> MessageT:
#         """
#         Get message with WRITE operation.
#
#         Parameters
#         ----------
#         data : Any
#             data to writing.
#
#         Returns
#         -------
#         MessageT
#             message instance.
#         """
#         return self._reg.write(self._pat, data)
#
#
# class FormatsMapABC(ABC, Generic[PatternsMapT, RegistersMapT, MessageT]):
#     """
#     Represents class with patterns and registers container.
#     """
#
#     def __init__(
#         self, patterns: PatternsMapT, registers: RegistersMapT
#     ) -> None:
#         self._patterns = patterns
#         self._registers = registers
#
#     @property
#     def patterns(self) -> PatternsMapT:
#         """
#         Returns
#         -------
#         PatternsMapT
#             patterns container.
#         """
#         return self._patterns
#
#     @property
#     def registers(self) -> RegistersMapT:
#         """
#         Returns
#         -------
#         RegistersMapT
#             registers container.
#         """
#         return self._registers
#
#     def __getitem__(self, name: str) -> RegisterParser[MessageT]:
#         """
#         Get register parser with `name` register.
#
#         Parameters
#         ----------
#         name : str
#             name of register.
#
#         Returns
#         -------
#         RegisterParser[MessageT]
#             register parser.
#         """
#         register = self._registers.get_register(name)
#         pattern = self._patterns[register.pattern]
#         return RegisterParser(pattern, register)

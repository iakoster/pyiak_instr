# from pathlib import Path
# from configparser import ConfigParser
# from typing import Any
#
# from src.pyiak_instr.store.bin.types import STRUCT_DATACLASS
# from src.pyiak_instr.communication.message.types import Message, MessagePattern
# from src.pyiak_instr.communication.format_map.types import (
#     FormatsMapABC,
#     PatternsMapABC,
#     RegisterABC,
#     RegistersMapABC,
# )
#
#
# class TIPatternsMap(PatternsMapABC[MessagePattern]):
#
#     _pattern_type = MessagePattern
#
#
# @STRUCT_DATACLASS
# class TIRegister(RegisterABC):
#     ...
#
#
# class TIRegistersMap(RegistersMapABC[TIRegister]):
#
#     _register_type = TIRegister
#
#
# class TIFormatsMap(FormatsMapABC[TIPatternsMap, TIRegistersMap, Message[Any]]):
#     ...

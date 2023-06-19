from src.pyiak_instr.store.bin import STRUCT_DATACLASS
from src.pyiak_instr.communication.format import Register, RegisterMap


@STRUCT_DATACLASS
class TIRegister(Register):
    ...


class TIRegisterMap(RegisterMap[TIRegister]):
    _register_type = TIRegister

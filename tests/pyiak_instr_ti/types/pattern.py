from src.pyiak_instr.types import (
    Pattern,
    SurPattern,
    EditableMixin,
)


class TIPattern(Pattern[dict], EditableMixin):

    _options = {"basic": dict}


class TISurPattern(SurPattern[dict, TIPattern]):

    _options = {"basic": dict}
    _sub_p_type = TIPattern

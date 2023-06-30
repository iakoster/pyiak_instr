from src.pyiak_instr.types import (
    Additions,
    Pattern,
    SurPattern,
    EditableMixin,
)


class TIPattern(Pattern[dict], EditableMixin):

    _options = {"basic": dict}


class TISurPattern(SurPattern[dict, TIPattern]):

    _options = {"basic": dict}
    _sub_p_type = TIPattern

    def _modify_additions(self, additions: Additions | None = None) -> Additions:
        add = super()._modify_additions(additions)
        add.current["subs"] = {
            n: p.get(add.lower(n)) for n, p in self._sub_p.items()
        }
        return add

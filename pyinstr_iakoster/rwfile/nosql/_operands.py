

__all__ = [
    "NoSqlOperand",
    "UNSET",
    "SET"
]


class NoSqlOperand(object):

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, **fields):
        return {self._name: {f: v for f, v in fields.items()}}


class NoSqlUnsetOperand(NoSqlOperand):

    def __init__(self):
        super(NoSqlUnsetOperand, self).__init__("$unset")

    def __call__(self, *fields: str):
        return super(NoSqlUnsetOperand, self)\
            .__call__(**{f: "" for f in fields})


class NoSqlSetOperand(NoSqlOperand):

    def __init__(self):
        super(NoSqlSetOperand, self).__init__("$set")


UNSET = NoSqlUnsetOperand()
SET = NoSqlSetOperand()

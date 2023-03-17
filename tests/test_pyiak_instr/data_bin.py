from src.pyiak_instr.core import Code
from src.pyiak_instr.store import BytesField, ContinuousBytesStorage


__all__ = [
    "get_cbs_one",
    "get_cbs_example",
    "get_cbs_one_infinite",
    "get_cbs_first_infinite",
    "get_cbs_middle_infinite",
    "get_cbs_last_infinite",
]


def get_cbs_one() -> ContinuousBytesStorage:
    return ContinuousBytesStorage(
        name="cbs_one",
        f0=BytesField(
            start=0,
            fmt=Code.I16,
            expected=2,
        )
    )


def get_cbs_example() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        name="cbs_example",
        f0=BytesField(
            start=0,
            fmt=Code.U16,
            expected=1,
        ),
        f1=BytesField(
            start=2,
            fmt=Code.U8,
            expected=2,
            order=Code.LITTLE_ENDIAN,
        ),
        f2=BytesField(
            start=4,
            fmt=Code.I8,
            expected=-1,
            order=Code.LITTLE_ENDIAN,
        ),
        f3=BytesField(
            start=-3,
            fmt=Code.U8,
            expected=2,
            order=Code.LITTLE_ENDIAN,
        ),
        f4=BytesField(
            start=-1,
            fmt=Code.I8,
            expected=1,
            order=Code.LITTLE_ENDIAN,
        )
    )
    object.__setattr__(cbs["f2"].fld, "_stop", -3)
    return cbs


def get_cbs_one_infinite() -> ContinuousBytesStorage:
    return ContinuousBytesStorage(
        name="cbs_one_infinite",
        f0=BytesField(
            start=0,
            fmt=Code.I8,
            expected=-1,
            order=Code.LITTLE_ENDIAN,
        )
    )


def get_cbs_first_infinite() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        name="cbs_first_infinite",
        f0=BytesField(
            start=0,
            fmt=Code.I8,
            expected=-1,
            order=Code.LITTLE_ENDIAN,
        ),
        f1=BytesField(
            start=-4,
            fmt=Code.U16,
            expected=1,
        ),
        f2=BytesField(
            start=-2,
            fmt=Code.U8,
            expected=2,
            order=Code.LITTLE_ENDIAN,
        ),
    )
    object.__setattr__(cbs["f0"].fld, "_stop", -4)
    return cbs


def get_cbs_middle_infinite() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        name="cbs_middle_infinite",
        f0=BytesField(
            start=0,
            fmt=Code.U16,
            expected=2,
            order=Code.LITTLE_ENDIAN,
        ),
        f1=BytesField(
            start=4,
            fmt=Code.U16,
            expected=-1,
        ),
        f2=BytesField(
            start=-2,
            fmt=Code.U8,
            expected=2,
            order=Code.LITTLE_ENDIAN,
        ),
    )
    object.__setattr__(cbs["f1"].fld, "_stop", -2)
    return cbs


def get_cbs_last_infinite() -> ContinuousBytesStorage:
    return ContinuousBytesStorage(
        name="cbs_last_infinite",
        f0=BytesField(
            start=0,
            fmt=Code.U16,
            order=Code.LITTLE_ENDIAN,
            expected=1,
        ),
        f1=BytesField(
            start=2,
            fmt=Code.U32,
            expected=-1,
        ),
    )


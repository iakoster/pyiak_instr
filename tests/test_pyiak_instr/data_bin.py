from src.pyiak_instr.core import Code
from src.pyiak_instr.store import BytesFieldStruct, ContinuousBytesStorage


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
        "cbs_one", dict(
            f0=BytesFieldStruct(
                start=0,
                fmt=Code.I16,
                bytes_expected=4,
            )
        )
    )


def get_cbs_example() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        "cbs_example", dict(
            f0=BytesFieldStruct(
                start=0,
                fmt=Code.U16,
                bytes_expected=2,
            ),
            f1=BytesFieldStruct(
                start=2,
                fmt=Code.U8,
                bytes_expected=2,
                order=Code.LITTLE_ENDIAN,
            ),
            f2=BytesFieldStruct(
                start=4,
                fmt=Code.I8,
                bytes_expected=-1,
                order=Code.LITTLE_ENDIAN,
            ),
            f3=BytesFieldStruct(
                start=-3,
                fmt=Code.U8,
                bytes_expected=2,
                order=Code.LITTLE_ENDIAN,
            ),
            f4=BytesFieldStruct(
                start=-1,
                fmt=Code.I8,
                bytes_expected=1,
                order=Code.LITTLE_ENDIAN,
            )
        )
    )
    object.__setattr__(cbs["f2"].struct, "_stop", -3)
    return cbs


def get_cbs_one_infinite() -> ContinuousBytesStorage:
    return ContinuousBytesStorage(
        "cbs_one_infinite", dict(
            f0=BytesFieldStruct(
                start=0,
                fmt=Code.I8,
                bytes_expected=-1,
                order=Code.LITTLE_ENDIAN,
            )
        )
    )


def get_cbs_first_infinite() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        "cbs_first_infinite", dict(
            f0=BytesFieldStruct(
                start=0,
                fmt=Code.I8,
                bytes_expected=-1,
                order=Code.LITTLE_ENDIAN,
            ),
            f1=BytesFieldStruct(
                start=-4,
                fmt=Code.U16,
                bytes_expected=2,
            ),
            f2=BytesFieldStruct(
                start=-2,
                fmt=Code.U8,
                bytes_expected=2,
                order=Code.LITTLE_ENDIAN,
            ),
        )
    )
    object.__setattr__(cbs["f0"].struct, "_stop", -4)
    return cbs


def get_cbs_middle_infinite() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        "cbs_middle_infinite", dict(
            f0=BytesFieldStruct(
                start=0,
                fmt=Code.U16,
                bytes_expected=4,
                order=Code.LITTLE_ENDIAN,
            ),
            f1=BytesFieldStruct(
                start=4,
                fmt=Code.U16,
                bytes_expected=-1,
            ),
            f2=BytesFieldStruct(
                start=-2,
                fmt=Code.U8,
                bytes_expected=2,
                order=Code.LITTLE_ENDIAN,
            ),
        )
    )
    object.__setattr__(cbs["f1"].struct, "_stop", -2)
    return cbs


def get_cbs_last_infinite() -> ContinuousBytesStorage:
    return ContinuousBytesStorage(
        "cbs_last_infinite", dict(
            f0=BytesFieldStruct(
                start=0,
                fmt=Code.U16,
                order=Code.LITTLE_ENDIAN,
                bytes_expected=2,
            ),
            f1=BytesFieldStruct(
                start=2,
                fmt=Code.U32,
                bytes_expected=-1,
            ),
        )
    )


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
            fmt="h",
            order=">",
            expected=2,
        )
    )


def get_cbs_example() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        name="cbs_example",
        f0=BytesField(
            start=0,
            fmt="H",
            order=">",
            expected=1,
        ),
        f1=BytesField(
            start=2,
            fmt="B",
            order="",
            expected=2,
        ),
        f2=BytesField(
            start=4,
            fmt="b",
            order="",
            expected=-1,
        ),
        f3=BytesField(
            start=-3,
            fmt="B",
            order="",
            expected=2,
        ),
        f4=BytesField(
            start=-1,
            fmt="b",
            order="",
            expected=1,
        )
    )
    object.__setattr__(cbs["f2"].fld, "_stop", -3)
    return cbs


def get_cbs_one_infinite() -> ContinuousBytesStorage:
    return ContinuousBytesStorage(
        name="cbs_one_infinite",
        f0=BytesField(
            start=0,
            fmt="b",
            order="",
            expected=-1,
        )
    )


def get_cbs_first_infinite() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        name="cbs_first_infinite",
        f0=BytesField(
            start=0,
            fmt="b",
            order="",
            expected=-1,
        ),
        f1=BytesField(
            start=-4,
            fmt="H",
            order=">",
            expected=1,
        ),
        f2=BytesField(
            start=-2,
            fmt="B",
            order="",
            expected=2,
        ),
    )
    object.__setattr__(cbs["f0"].fld, "_stop", -4)
    return cbs


def get_cbs_middle_infinite() -> ContinuousBytesStorage:
    cbs = ContinuousBytesStorage(
        name="cbs_middle_infinite",
        f0=BytesField(
            start=0,
            fmt="H",
            order="",
            expected=2,
        ),
        f1=BytesField(
            start=4,
            fmt="H",
            order=">",
            expected=-1,
        ),
        f2=BytesField(
            start=-2,
            fmt="B",
            order="",
            expected=2,
        ),
    )
    object.__setattr__(cbs["f1"].fld, "_stop", -2)
    return cbs


def get_cbs_last_infinite() -> ContinuousBytesStorage:
    return ContinuousBytesStorage(
        name="cbs_last_infinite",
        f0=BytesField(
            start=0,
            fmt="H",
            order="",
            expected=1,
        ),
        f1=BytesField(
            start=2,
            fmt="I",
            order=">",
            expected=-1,
        ),
    )


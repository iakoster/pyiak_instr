from src.pyiak_instr.store import BytesField, ContinuousBytesStorage


__all__ = [
    "CBS_ONE",

]


CBS_ONE = ContinuousBytesStorage(
    f0=BytesField(
        start=0,
        fmt="h",
        order=">",
        expected=2,
    )
)

CBS_EXAMPLE = ContinuousBytesStorage(

)

CBS_ONE_INFINITE = ContinuousBytesStorage(
    f0=BytesField(
        start=0,
        fmt="b",
        order="",
        expected=-1,
    )
)

CBS_FIRST_INFINITE = ContinuousBytesStorage(
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
        expected=-1,
    ),
    f2=BytesField(

    )
)

CBS_MIDDLE_INFINITE = ContinuousBytesStorage(

)

CBS_LAST_INFINITE = ContinuousBytesStorage(

)


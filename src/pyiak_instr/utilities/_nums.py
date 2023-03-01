"""Private module of ``pyiak_instr.utilities`` with functions with nums"""
import string


__all__ = ["num_sign", "to_base"]


def num_sign(value: int | float, pos_zero: bool = False) -> int:
    """
    Get sign of the value.

    Returns +1 if 'value' is positive, -1 - if negative and
    0 - if 'value' is zero.

    Parameters
    ----------
    value: int or float
        value to define.
    pos_zero: bool
        (positive zero) if True than function returns +1 if 'value' == 0.

    Returns
    -------
    int
        sign of a number or zero if value is zero if 'pos_zero' is False.
    """
    sign = (value > 0) - (value < 0)
    if sign:
        return sign
    if pos_zero:
        return 1
    return 0


def to_base(value: int, base: int) -> str:
    """
    Convert integer with base 10 to selected base.

    The inverse conversion can be performed with the function int
    (e.g. int(value, base)).

    Parameters
    ----------
    value: int
        value for converting.
    base: int
        integer base.

    Returns
    -------
    str
        value with selected base.

    Raises
    ------
    ValueError
        if base not in range [2; 36].
    """
    if not 2 <= base <= 36:
        raise ValueError("base must be in range [2; 36]")

    if value == 0:
        return "0"
    digits = string.digits + string.ascii_lowercase
    sign = num_sign(value)
    value = abs(value)
    val_digits = ""

    while value:
        value, mod = divmod(value, base)
        val_digits += digits[mod]
    if sign < 0:
        val_digits += "-"

    return val_digits[::-1]

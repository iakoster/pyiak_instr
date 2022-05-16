import string


def num_sign(value: int | float, pos_zero: bool = False) -> int:
    """
    Get sign of the value.

    Returns +1 if value is positive, -1 if negative and 0 if value is zero.

    Parameters
    ----------
    value: int or float
        value to define.
    pos_zero: bool
        (positive zero) if True than function returns +1 if value == 0.

    Returns
    -------
    int
        sign of a number or zero if value is zero if pos_zero is False.
    """
    sign = (value > 0) - (value < 0)
    if pos_zero and sign == 0:
        return 1
    return sign


def to_base(val: int, base: int) -> str:
    """
    Convert integer with base 10 to selected base.

    Parameters
    ----------
    val: int
        value for convertig.
    base: int
        integer base.

    Returns
    -------
    str
        value with selected base.

    Raises
    ------
    ValueError
        if base not in range [2; 36]
    """
    if not 1 < base < 37:
        raise ValueError("base must be in range [2; 36]")

    if val == 0:
        return "0"
    digits = string.digits + string.ascii_letters
    sign = num_sign(val)
    val = abs(val)
    val_digits = []

    while val:
        val, mod = divmod(val, base)
        val_digits.append(digits[mod])
    if sign < 0:
        val_digits.append('-')
    val_digits.reverse()

    return ''.join(val_digits)

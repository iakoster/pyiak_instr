import string


__all__ = ["num_sign", "to_base", "from_base"]


def num_sign(value: int | float, pos_zero: bool = False) -> int:
    """
    Get sign of the value.

    Returns +1 if value is positive, -1 - if negative and
    0 -- if value is zero.

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


def to_base(value: int, base: int) -> str:
    """
    Convert integer with base 10 to selected base.

    Parameters
    ----------
    value: int
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
        if base not in range [2; 36].
    """
    if not 1 < base < 37:
        raise ValueError("base must be in range [2; 36]")

    if value == 0:
        return "0"
    digits = string.digits + string.ascii_lowercase
    sign = num_sign(value)
    value = abs(value)
    val_digits = []

    while value:
        value, mod = divmod(value, base)
        val_digits.append(digits[mod])
    if sign < 0:
        val_digits.append('-')
    val_digits.reverse()

    return ''.join(val_digits)


def from_base(value: str, base: int) -> int:
    """
    Convert integer with some base 10 to integer with base 10.

    Parameters
    ----------
    value: str
        value for converting.
    base: int
        base of the value.

    Returns
    -------
    int
        converted value.

    Raises
    ------
    ValueError
        if base not in range [2; 36].
    """
    if not 1 < base < 37:
        raise ValueError("base must be in range [2; 36]")

    if value == "0":
        return 0
    if value[0] == "-":
        negative = True
        value = value[1:].lower()
    else:
        negative = False
        value = value.lower()

    digits = string.digits + string.ascii_lowercase
    int_value = 0

    for i_digit, digit in enumerate(value[::-1]):
        int_value += digits.index(digit) * base ** i_digit

    if negative:
        return ~int_value + 1
    return int_value

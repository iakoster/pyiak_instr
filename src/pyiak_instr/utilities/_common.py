"""Private module of ``pyiak_instr.utilities`` with common functions"""
from typing import Any


__all__ = ["split_complex_dict"]


def split_complex_dict(
    complex_dict: dict[str, Any], sep: str = "__", without_sep: str = "raise"
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Split dictionary to nested dictionaries (subdictionaries) by `sep`.

    Parameters
    ----------
    complex_dict: dict[str, Any]
        dictionary for splitting.
    sep: str, default='__'
        separator of nested dictionaries.
    without_sep: str, default='raise'
        behavior if key without sep is detected: 'raise' - raise error,
        'other' - put that keys to other dictionary.

    Returns
    -------
    tuple[dict[str, dict[str, Any]], dict[str, Any]]
        two dictionaries:
            nested dictionaries separated by `sep`;
            dictionary with keys without `sep`. It will be empty if there is
            no keys without `sep` and `without_sep`='other'.

    Raises
    ------
    ValueError
        if keyword argument `without_sep` not in {'raise', 'other'}.
    KeyError
        if `without_sep' is equal to 'raise' and key does not have 'sep'.
    """
    if without_sep not in {"raise", "other"}:
        raise ValueError(
            f"invalid attribute 'without_sep': '{without_sep}' not in "
            "{'raise', 'other'}"
        )

    result: dict[str, dict[str, Any]]
    result, wo_sep_dict = {}, {}
    for key, value in complex_dict.items():
        if without_sep == "raise" and sep not in key:
            raise KeyError(f"key '{key}' does not have separator '{sep}'")
        if sep not in key:
            wo_sep_dict[key] = value
            continue

        sub_keys = key.split(sep)
        sub_dict = result

        for i_sub_key, sub_key in enumerate(sub_keys):
            if i_sub_key == len(sub_keys) - 1:
                break

            if sub_key not in sub_dict:
                sub_dict[sub_key] = {}
            sub_dict = sub_dict[sub_key]

        sub_dict[sub_keys[-1]] = value

    return result, wo_sep_dict

"""Private module of ``pyiak_instr.utilities``"""
from typing import Any, Literal

from ..exceptions import NotAmongTheOptions


__all__ = ["split_complex_dict"]


def split_complex_dict(
    complex_dict: dict[str, Any],
    sep: str = "__",
    without_sep: Literal["raise", "ignore", "other"] = "raise",
    split_level: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Split dictionary to nested dictionaries (subdictionaries) by `sep`.

    Parameters
    ----------
    complex_dict: dict[str, Any]
        dictionary for splitting.
    sep: str, default='__'
        separator of nested dictionaries.
    without_sep: Literal['raise', 'ignore', 'other'], default='raise'
        behavior if key without `sep` is detected:
            * 'raise' - raise error;
            * 'ignore' - ignore that keys;
            * 'other' - put that keys to other dictionary.
    split_level: int, default=0
        level to which the complex vocabulary is divided. For example, if
        the value is 1, the keys will be divided by `sep` up to its first
        occurrence ({key_1__key_2__key_3: ...} ->
        {key_1: {key_2__key_3: ...}}).

    Returns
    -------
    tuple[dict[str, dict[str, Any]], dict[str, Any]]
        two dictionaries:
            * nested dictionaries separated by `sep`;
            * dictionary with keys without `sep`. It will be empty if there is
            no keys without `sep` and `without_sep`='other'.

    Raises
    ------
    NotAmongTheOptions
        if keyword argument `without_sep` not in {"raise", "ignore", "other"}.
    KeyError
        if `without_sep' is equal to 'raise' and key does not have 'sep'.
    """
    if without_sep not in {"raise", "ignore", "other"}:
        raise NotAmongTheOptions(
            "without_sep", without_sep, {"raise", "ignore", "other"}
        )

    result: dict[str, dict[str, Any]]
    result, wo_sep = {}, {}
    for key, value in complex_dict.items():
        if sep not in key:
            if without_sep == "raise":
                raise KeyError(f"key '{key}' does not have separator '{sep}'")
            if without_sep == "other":
                wo_sep[key] = value
            continue

        sub_keys = key.split(sep)
        sub_dict = result

        i_sub_key = 0
        while i_sub_key < len(sub_keys) - 1:
            if 0 < split_level == i_sub_key:
                break
            sub_key = sub_keys[i_sub_key]
            if sub_key not in sub_dict:
                sub_dict[sub_key] = {}
            sub_dict = sub_dict[sub_key]
            i_sub_key += 1

        sub_dict[sep.join(sub_keys[i_sub_key:])] = value

    return result, wo_sep

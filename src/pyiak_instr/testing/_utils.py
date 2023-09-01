"""Private module of ``pyiak_instr.testing``."""
import inspect
import types
from typing import Any


__all__ = ["get_members", "get_object_attrs"]


# pylint: disable=too-many-branches
def get_members(
    obj: type | object, pass_attr: list[str]
) -> list[tuple[str, Any]]:
    """
    Modified `_getmembers` function from `inspect`.

    Parameters
    ----------
    obj : type | object
        object from which members will be extracted.
    pass_attr : list[str]
        attributes which will be not added to result.

    Returns
    -------
    list[tuple[str, Any]]
        all `obj` members.

    See Also
    --------
    inspect._getmembers : source function.
    """
    results = []
    processed = set()
    names = dir(obj)

    if inspect.isclass(obj):
        mro = (obj,) + inspect.getmro(obj)
        try:
            for base in obj.__bases__:
                for k, v in base.__dict__.items():
                    if isinstance(v, types.DynamicClassAttribute):
                        names.append(k)
        except AttributeError:
            pass

    else:
        mro = ()

    for key in names:
        if key in pass_attr:
            continue

        try:
            value = getattr(obj, key)
            if key in processed:
                raise AttributeError

        except AttributeError:
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                continue

        results.append((key, value))
        processed.add(key)

    results.sort(key=lambda pair: pair[0])
    return results


def get_object_attrs(
    obj: type | object, wo_attrs: list[str] = None, wo_consts: bool = True
) -> list[str]:
    """
    Get not callable attributes and properties from class.

    Parameters
    ----------
    obj : type | object
        object from which attributes will be extracted.
    wo_attrs : list[str], default=None
        attributes which will be not added to result.
    wo_consts : bool, default=True
        indicates that constants will not as a result. An attribute whose
        name is written in upper case is considered to be a constant.

    Returns
    -------
    list[str]
        properties of a `obj`.
    """
    if wo_attrs is None:
        wo_attrs = []

    # pylint: disable=missing-return-doc
    def is_attribute(name: str, func: Any) -> bool:
        return (  # is not callable
            not (inspect.ismethod(func) or inspect.isfunction(func))
        ) and (  # is correct name
            not (name.startswith("_") or wo_consts and name.isupper())
        )

    props = [n for n, m in get_members(obj, wo_attrs) if is_attribute(n, m)]
    for attr in wo_attrs:
        if attr in props:
            props.pop(props.index(attr))
    return props

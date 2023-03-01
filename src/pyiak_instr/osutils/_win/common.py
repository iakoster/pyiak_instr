"""Private module of ``pyiak_instr.osutils`` with Windows functions."""
import os
import stat
from subprocess import call


__all__ = [
    "hide_path",
    "unhide_path",
    "is_hidden_path",
]


def hide_path(path: os.PathLike[str]) -> None:
    """
    Hide file or directory.

    Parameters
    ----------
    path: PathLike[str]
        path to the file or directory.

    Raises
    ------
    FileNotFoundError
        if path not exists.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("path not found")
    call(["attrib", "+H", path])


def unhide_path(path: os.PathLike[str]) -> None:
    """
    Unhide file or directory.

    Parameters
    ----------
    path: PathLike[str]
        path to the file or directory

    Raises
    ------
    FileNotFoundError
        if path not exists.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("path not found")
    call(["attrib", "-H", path])


def is_hidden_path(path: os.PathLike[str]) -> bool:
    """
    Check that path is hidden.

    Parameters
    ----------
    path: PathLike[str]
        path to the file or directory

    Returns
    -------
    bool
        indicates that directory or file is hidden.

    Raises
    ------
    FileNotFoundError
        if path not exists.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("path not found")
    return bool(os.stat(path).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)

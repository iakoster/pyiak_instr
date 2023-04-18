from pathlib import Path


__all__ = [
    "hide_path",
    "unhide_path",
    "is_hidden_path",
]


def hide_path(path: Path) -> Path:
    """
    Hide file or directory.

    Do not do anything on Linux.

    Parameters
    ----------
    path: Path
        path to the file or directory.

    Returns
    -------
    Path
        path to the file or directory.

    Raises
    ------
    FileExistsError
        if path not exists.
    """
    if not path.exists():
        raise FileExistsError("path not exists")
    return path


def unhide_path(path: Path) -> Path:
    """
    Unhide file or directory.

    Do not do anything on Linux.

    Parameters
    ----------
    path: Path
        path to the file or directory

    Returns
    -------
    Path
        path to the file or directory.

    Raises
    ------
    FileExistsError
        if path not exists.
    """
    if not path.exists():
        raise FileExistsError("path not exists")
    return path


def is_hidden_path(path: Path) -> bool:
    """
    Check that path is hidden.

    Parameters
    ----------
    path: Path
        path to the file or directory

    Returns
    -------
    bool
        indicates that directory or file is hidden.

    Raises
    ------
    FileExistsError
        if path not exists.
    """
    if not path.exists():
        raise FileExistsError("path not exists")
    return path.name[0] == "."

from re import Pattern
from pathlib import Path

from ._exception import FilepathPatternError


__all__ = ['match_filename', 'if_str2path', 'create_dir_if_not_exists']


def match_filename(pattern: Pattern, path: Path):
    """
    Checks the path by pattern via match method.

    Parameters
    ----------
    pattern: re.Pattern
        file name pattern.
    path: Path
        path to the file.

    Raises
    ------
    FilepathPatternError:
        if the pattern is not match in the path (file name).
    """
    if pattern.match(path.name) is None:
        raise FilepathPatternError(pattern.pattern, path)


def if_str2path(path: Path | str) -> Path:
    """
    Convert the path-like str into a Path instance or
    return path as is.

    Parameters
    ----------
    path: Path or path-like str
        some path.

    Returns
    -------
    Path
        some path in Path instance
    """
    if isinstance(path, str):
        path = Path(path)
    return path


def create_dir_if_not_exists(path: Path, to_file: bool = True):
    """
    Create a directory if it does not exists.

    Parameters
    ----------
    path: Path
        path to a file or a directory.
    to_file: bool, default=True
        indicates that the path points to the file.
    """
    if to_file:
        path = path.parent
    if not path.exists():
        path.mkdir(parents=True)

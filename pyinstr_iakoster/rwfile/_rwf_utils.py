from re import Pattern
from pathlib import Path

from ._rwf_exception import FilepathPatternError


__all__ = ['match_filename', 'if_str2path', 'create_dir_if_not_exists']


def match_filename(pattern: Pattern, path: Path):
    """
    checks the path by pattern using the match method.

    :param pattern: file name pattern
    :param path: path to the file
    :raise FilepathPatternError: if the pattern is not match
        in the path (file name)
    """
    if pattern.match(path.name) is None:
        raise FilepathPatternError(pattern.pattern, path)


def if_str2path(path: Path | str) -> Path:
    """
    convert the path into a Path instance if
    the path is a string

    :param path: some path
    :return: path
    :rtype: Path
    """
    if isinstance(path, str):
        path = Path(path)
    return path


def create_dir_if_not_exists(path: Path, to_file: bool = True):
    """
    create a directory if it does not exists

    :param path: path to a file or a directory
    :param to_file: indicates that the path points to the file or not
    """
    if to_file:
        path = path.parent
    if not path.exists():
        path.mkdir(parents=True)

from re import Pattern
from pathlib import Path

from ._rwf_exception import FilepathPatternError


__all__ = ['check_filename', 'if_str2path', 'create_dir_if_not_exists']


def check_filename(pattern: Pattern, path: Path):
    if pattern.match(path.name) is None:
        raise FilepathPatternError(pattern, path)


def if_str2path(path):
    if isinstance(path, str):
        path = Path(path)
    return path


def create_dir_if_not_exists(path: Path, to_file: bool = True):
    if to_file:
        path = path.parent
    if not path.exists():
        path.mkdir(parents=True)

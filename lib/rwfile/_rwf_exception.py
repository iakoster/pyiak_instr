import re
from pathlib import Path


__all__ = ['FilepathPatternError']


class FilepathPatternError(Exception):

    def __init__(self, pattern: re.Pattern, filepath: Path | str):
        self.message = (
            'The path does not lead to '
            '%r file' % pattern.pattern)
        Exception.__init__(self, self.message)
        if isinstance(filepath, str):
            filepath = Path(filepath)
        self.args = (self.message, filepath)

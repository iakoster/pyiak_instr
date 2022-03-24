from pathlib import Path


__all__ = ['FilepathPatternError']


class FilepathPatternError(Exception):
    """
    Raised when filepath is wrong by some pattern
    """

    def __init__(self, pattern: str, filepath: Path | str):
        self.message = (
            'The path does not lead to '
            '%r file' % pattern)
        Exception.__init__(self, self.message)
        if isinstance(filepath, str):
            filepath = Path(filepath)
        self.args = (self.message, filepath)

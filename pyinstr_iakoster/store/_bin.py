from pathlib import Path

from ..rwfile import RWBin


__all__ = ["BinaryData"]


class BinaryData(object):

    def __init__(self, content: bytes) -> None:
        self._content = content

    @classmethod
    def read(cls, filepath: Path | str) -> "BinaryData":
        with RWBin(filepath) as rwb:
            return cls(rwb.read_all())

    @property
    def content(self) -> bytes:
        return self._content

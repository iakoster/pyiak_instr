from pathlib import Path

from ._mess import FieldSetter, Message


class PackageFormatBase(object):

    def __init__(self):
        self._setters = {}

    @property
    def setters(self) -> dict[str, FieldSetter]:
        return self._setters


class PackageFormat(PackageFormatBase):

    def __init__(self):
        PackageFormatBase.__init__(self)

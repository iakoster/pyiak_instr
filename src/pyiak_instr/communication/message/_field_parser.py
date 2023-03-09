"""Private module of ``pyiak_instr.communication.message`` with field
parsers."""
from ._field import MessageField
from ...store import BytesFieldParser


__all__ = ["MessageFieldParser"]


class MessageFieldParser(BytesFieldParser):
    """
    Represents parser for work with message field content.
    """

    _f: MessageField  # todo: typing

    @property
    def fld(self) -> MessageField:
        """
        Returns
        -------
        MessageField
            field instance
        """
        return self._f

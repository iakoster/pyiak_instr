import unittest

from src.pyiak_instr.communication.message import (
    Message,
    MessageFieldStruct,
    MessageStruct,
)

from ....utils import validate_object


class TestMessage(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            Message(storage=MessageStruct(fields={
                "f0": MessageFieldStruct(name="f0"),
            })),
            dst=None,
            has_pattern=False,
            src=None,
            wo_attrs=["get", "has", "struct"]
        )

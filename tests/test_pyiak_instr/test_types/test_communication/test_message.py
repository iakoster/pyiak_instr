import unittest

from src.pyiak_instr.store import BytesFieldStruct
from src.pyiak_instr.types.store import STRUCT_DATACLASS
from src.pyiak_instr.types.communication import (
    MessageABC,
    MessageFieldABC,
    MessageGetParserABC,
    MessageHasParserABC,
)

from ....utils import validate_object


@STRUCT_DATACLASS
class TIMessageFieldStruct(BytesFieldStruct):
    ...


class TIMessageField(MessageFieldABC["TIMessage", TIMessageFieldStruct]):
    ...


class TIMessageGetParser(MessageGetParserABC["TIMessage", TIMessageField]):
    ...


class TIMessageHasParser(MessageHasParserABC[TIMessageField]):
    ...


class TIMessage(
    MessageABC[
        TIMessageField,
        TIMessageFieldStruct,
        TIMessageGetParser,
        TIMessageHasParser,
        str,
    ]
):

    _get_parser = TIMessageGetParser
    _has_parser = TIMessageHasParser
    _struct_field = {TIMessageFieldStruct: TIMessageField}


class TestMessageABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessage(f0=TIMessageFieldStruct()),
            content=b"",
            divisible=False,
            dst=None,
            is_dynamic=True,
            minimum_size=0,
            mtu=1500,
            name="std",
            src=None,
            src_dst=(None, None),
            wo_attrs=["get", "has"]
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIMessage(
                divisible=True,
                mtu=5,
                f0=TIMessageFieldStruct(bytes_expected=10),
            )
        self.assertEqual(
            "MTU cannot be less than the minimum size", exc.exception.args[0]
        )

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

    @property
    def basic(self) -> TIMessageField:
        return self(TIMessageField)


class TIMessageHasParser(MessageHasParserABC[TIMessageField]):

    @property
    def basic(self) -> bool:
        return self(TIMessageField)


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

    def test_get_has(self) -> None:
        instance = TIMessage(
            f0=TIMessageFieldStruct(stop=5),
            f1=TIMessageFieldStruct(start=5),
        )

        with self.subTest(test="get basic"):
            self.assertEqual("f0", instance.get.basic.name)

        with self.subTest(test="has basic"):
            self.assertTrue(instance.has.basic)

        with self.subTest(test="hasn't other"):
            self.assertFalse(instance.has(MessageFieldABC))

    def test_get_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIMessage("test", f0=TIMessageFieldStruct()).get(MessageFieldABC)
        self.assertEqual(
            "MessageFieldABC instance is not found", exc.exception.args[0]
        )

    def test_src_dst(self) -> None:
        instance = TIMessage(f0=TIMessageFieldStruct())

        self.assertTupleEqual((None, None), instance.src_dst)
        instance.src_dst = None, "test"
        self.assertTupleEqual((None, "test"), instance.src_dst)
        instance.src = "alal"
        self.assertEqual("alal", instance.src)
        instance.dst = "test/2"
        self.assertEqual("test/2", instance.dst)

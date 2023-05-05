import unittest
from typing import Generator, Self

from src.pyiak_instr.core import Code
from src.pyiak_instr.store import BytesFieldStruct
from src.pyiak_instr.types.store import STRUCT_DATACLASS
from src.pyiak_instr.types.communication import (
    MessageABC,
    MessageFieldABC,
    MessageFieldPatternABC,
    MessageGetParserABC,
    MessageHasParserABC,
    MessagePatternABC,
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
        "TIMessagePatternABC",
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

    def split(self) -> Generator[Self, None, None]:
        raise NotImplementedError()


class TIMessageFieldPatternABC(MessageFieldPatternABC):

    _options = {"basic": TIMessageFieldStruct}

    @staticmethod
    def get_bytesize(fmt: Code) -> int:
        if fmt is Code.U8:
            return 1
        if fmt is Code.U16:
            return 2
        raise ValueError(f"invalid fmt: {repr(fmt)}")


class TIMessagePatternABC(MessagePatternABC):

    _options = {"basic": TIMessage}



class TestMessageABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessage({"f0": TIMessageFieldStruct()}),
            content=b"",
            divisible=False,
            dst=None,
            has_pattern=False,
            is_dynamic=True,
            minimum_size=0,
            mtu=1500,
            name="std",
            src=None,
            src_dst=(None, None),
            wo_attrs=["get", "has"]
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="divisible without dynamic field"):
            with self.assertRaises(TypeError) as exc:
                TIMessage(
                    {"f0": TIMessageFieldStruct(bytes_expected=1)},
                    divisible=True,
                )
            self.assertEqual(
                "TIMessage can not be divided because it does not have "
                "a dynamic field",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid mtu"):
            with self.assertRaises(ValueError) as exc:
                TIMessage(
                    {
                        "f0": TIMessageFieldStruct(bytes_expected=10),
                        "f1": TIMessageFieldStruct(fmt=Code.U16, start=10),
                    },
                    divisible=True,
                    mtu=5,
                )
            self.assertEqual(
                "MTU value does not allow you to split the message if "
                "necessary. The minimum MTU is 12 (got 5)",
                exc.exception.args[0],
            )

    def test_get_has(self) -> None:
        instance = TIMessage(dict(
            f0=TIMessageFieldStruct(stop=5),
            f1=TIMessageFieldStruct(start=5),
        ))

        with self.subTest(test="get basic"):
            self.assertEqual("f0", instance.get.basic.name)

        with self.subTest(test="has basic"):
            self.assertTrue(instance.has.basic)

        with self.subTest(test="hasn't other"):
            self.assertFalse(instance.has(MessageFieldABC))

    def test_get_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIMessage(
                {"f0": TIMessageFieldStruct()}, "test"
            ).get(MessageFieldABC)
        self.assertEqual(
            "MessageFieldABC instance is not found", exc.exception.args[0]
        )

    def test_src_dst(self) -> None:
        instance = TIMessage({"f0": TIMessageFieldStruct()})

        self.assertTupleEqual((None, None), instance.src_dst)
        instance.src_dst = None, "test"
        self.assertTupleEqual((None, "test"), instance.src_dst)
        instance.src = "alal"
        self.assertEqual("alal", instance.src)
        instance.dst = "test/2"
        self.assertEqual("test/2", instance.dst)


class TestMessagePatternABC(unittest.TestCase):

    def test_get(self) -> None:
        pattern = TIMessagePatternABC("basic", "test").configure(
            f0=TIMessageFieldPatternABC("basic", bytes_expected=1),
            f1=TIMessageFieldPatternABC(
                "basic", bytes_expected=2, fmt=Code.U16
            ),
            f2=TIMessageFieldPatternABC("basic", bytes_expected=0),
            f3=TIMessageFieldPatternABC("basic", bytes_expected=2),
            f4=TIMessageFieldPatternABC("basic", bytes_expected=4, fmt=Code.U16),
        )
        res = pattern.get()

        validate_object(
            self,
            res,
            content=b"",
            divisible=False,
            dst=None,
            has_pattern=True,
            is_dynamic=True,
            minimum_size=9,
            mtu=1500,
            name="test",
            src=None,
            src_dst=(None, None),
            pattern=pattern,
            wo_attrs=["get", "has"]
        )

        for field, pars in dict(
            f0=dict(
                fmt=Code.U8,
                slice_=slice(0, 1),
                words_expected=1,
            ),
            f1=dict(
                fmt=Code.U16,
                slice_=slice(1, 3),
                words_expected=1,
            ),
            f2=dict(
                fmt=Code.U8,
                slice_=slice(3, -6),
                words_expected=0,
            ),
            f3=dict(
                fmt=Code.U8,
                slice_=slice(-6, -4),
                words_expected=2,
            ),
            f4=dict(
                fmt=Code.U16,
                slice_=slice(-4, None),
                words_expected=2,
            ),
        ).items():
            with self.subTest(field=field):
                validate_object(
                    self,
                    res[field].struct,
                    **pars,
                    wo_attrs=[
                        "bytes_expected",
                        "default",
                        "has_default",
                        "is_dynamic",
                        "order",
                        "start",
                        "stop",
                        "word_bytesize",
                    ]
                )

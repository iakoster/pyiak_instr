import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.communication.message import (
    MessagePattern,
    MessageFieldStructPattern,
    MessageStructPattern,
)

from ....utils import validate_object


class TestMessageFieldStructPattern(unittest.TestCase):

    def test_init(self) -> None:
        pattern = MessageFieldStructPattern.basic()
        validate_object(
            self,
            pattern,
            typename="basic",
            size=0,
            is_dynamic=True,
        )
        self.assertDictEqual(
            dict(
                bytes_expected=0,
                default=b"",
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                typename="basic",
            ),
            pattern.__init_kwargs__(),
        )


class TestMessageStructPattern(unittest.TestCase):

    def test_init(self) -> None:
        obj = MessageStructPattern.basic()
        validate_object(
            self,
            obj,
            typename="basic",
            sub_pattern_names=[],
        )
        self.assertDictEqual(
            dict(
                divisible=False,
                mtu=1500,
                typename="basic",
            ),
            obj.__init_kwargs__(),
        )


class TestMessagePattern(unittest.TestCase):

    def test_init(self) -> None:
        obj = MessagePattern.basic()
        validate_object(
            self,
            obj,
            typename="basic",
            sub_pattern_names=[],
        )
        self.assertDictEqual(
            dict(typename="basic"),
            obj.__init_kwargs__(),
        )

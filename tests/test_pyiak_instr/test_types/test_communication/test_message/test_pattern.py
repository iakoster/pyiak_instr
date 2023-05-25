import unittest

from src.pyiak_instr.core import Code

from .....utils import validate_object, get_object_attrs
from .ti import (
    TIMessageFieldStruct,
    TISingleMessageFieldStruct,
    TIStaticMessageFieldStruct,
    TIAddressMessageFieldStruct,
    TICrcMessageFieldStruct,
    TIDataMessageFieldStruct,
    TIDataLengthMessageFieldStruct,
    TIIdMessageFieldStruct,
    TIOperationMessageFieldStruct,
    TIResponseMessageFieldStruct,
    TIMessageStruct,
    TIMessage,
    TIMessageFieldStructPattern,
    TIMessageStructPattern,
    TIMessagePattern,
)


class TestMessageFieldStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessageFieldStructPattern(typename="id"),
            typename="id",
            is_dynamic=True,
            size=1,
        )

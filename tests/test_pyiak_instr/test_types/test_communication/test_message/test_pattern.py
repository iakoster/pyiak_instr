import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.types import SubPatternAdditions

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
            size=0,
        )

    def test_get(self) -> None:
        validate_object(
            self,
            TIMessageFieldStructPattern(
                typename="crc", default=b"aa"
            ).get(fmt=Code.U16),
            has_default=True,
            default=b"aa",
            word_bytesize=2,
            wo_fields=set(),
            name="",
            slice_=slice(0, 2),
            start=0,
            order=Code.BIG_ENDIAN,
            words_expected=1,
            poly=0x1021,
            bytes_expected=2,
            init=0,
            fmt=Code.U16,
            is_dynamic=False,
            stop=2,
            wo_attrs=["encoder"],
        )


class TestMessageStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessageStructPattern(typename="basic").configure(
                f0=TIMessageFieldStructPattern(
                    typename="static", default=b"a"
                ),
                f1=TIMessageFieldStructPattern(typename="data"),
            ),
            typename="basic",
        )

    def test_get(self) -> None:
        pattern = TIMessageStructPattern(
            typename="basic", divisible=True
        ).configure(
            f0=TIMessageFieldStructPattern(
                typename="static", default=b"a"
            ),
            f1=TIMessageFieldStructPattern(typename="data"),
        )
        msg = pattern.get(
            mtu=30,
            sub_additions=SubPatternAdditions().update_additions(
                "f1", fmt=Code.U16
            ),
        )

        validate_object(
            self,
            msg,
            is_dynamic=True,
            mtu=30,
            minimum_size=1,
            name="std",
            dynamic_field_name="f1",
            divisible=True,
            wo_attrs=["has", "get", "fields"],
        )
        validate_object(
            self,
            msg["f1"],
            has_default=False,
            default=b"",
            word_bytesize=2,
            name="f1",
            slice_=slice(0, None),
            start=0,
            order=Code.BIG_ENDIAN,
            words_expected=0,
            bytes_expected=0,
            fmt=Code.U16,
            is_dynamic=True,
            stop=None,
            wo_attrs=["encoder"],
        )


class TestMessagePatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessagePattern(typename="basic").configure(
                s0=TIMessageStructPattern(typename="basic").configure(
                    f0=TIMessageFieldStructPattern(
                        typename="static", default=b"a"
                    ),
                    f1=TIMessageFieldStructPattern(typename="data"),
                ),
            ),
            typename="basic",
        )

    def test_get(self) -> None:
        pattern = TIMessagePattern(typename="basic").configure(
            s0=TIMessageStructPattern(
                typename="basic", divisible=True
            ).configure(
                f0=TIMessageFieldStructPattern(
                    typename="static", default=b"a"
                ),
                f1=TIMessageFieldStructPattern(typename="data"),
            ),
        )
        msg = pattern.get(
            sub_additions=SubPatternAdditions().update_additions(
                "s0", mtu=30,
            ).set_next_additions(
                s0=SubPatternAdditions().update_additions(
                    "f1", fmt=Code.U16
                )
            ),
        )

        validate_object(
            self,
            msg,
            has_pattern=True,
            dst=None,
            src=None,
            wo_attrs=["pattern", "struct", "has", "get"],
        )
        validate_object(
            self,
            msg.struct,
            is_dynamic=True,
            mtu=30,
            minimum_size=1,
            name="s0",
            dynamic_field_name="f1",
            divisible=True,
            wo_attrs=["has", "get", "fields"],
        )
        validate_object(
            self,
            msg.struct["f1"],
            has_default=False,
            default=b"",
            word_bytesize=2,
            name="f1",
            slice_=slice(0, None),
            start=0,
            order=Code.BIG_ENDIAN,
            words_expected=0,
            bytes_expected=0,
            fmt=Code.U16,
            is_dynamic=True,
            stop=None,
            wo_attrs=["encoder"],
        )

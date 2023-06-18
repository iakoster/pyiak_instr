import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.types import SubPatternAdditions
from src.pyiak_instr.exceptions import NotAmongTheOptions

from .....utils import validate_object, get_object_attrs
from tests.pyiak_instr_ti.communication import (
    TIFieldPattern,
    TIStructPattern,
    TIMessagePattern,
)


class TestMessageFieldStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIFieldPattern(typename="id"),
            typename="id",
            is_dynamic=True,
            size=0,
            direction=Code.ANY,
        )

    def test_init_specific(self) -> None:
        cases = dict(
            basic=dict(
                typename="basic",
                direction=Code.ANY,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=0,
                default=b"",
            ),
            static=dict(
                typename="static",
                direction=Code.ANY,
                bytes_expected=1,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                default=b"\x00",
            ),
            address=dict(
                typename="address",
                direction=Code.ANY,
                bytes_expected=1,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                behaviour=Code.DMA,
                units=Code.WORDS,
                default=b"",
            ),
            crc=dict(
                typename="crc",
                direction=Code.ANY,
                bytes_expected=2,
                fmt=Code.U16,
                order=Code.BIG_ENDIAN,
                poly=0x1021,
                init=0,
                default=b"",
                fill_value=b"\x00",
                wo_fields=set(),
            ),
            data=dict(
                typename="data",
                direction=Code.ANY,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=0,
                default=b"",
            ),
            dynamic_length=dict(
                typename="dynamic_length",
                direction=Code.ANY,
                bytes_expected=1,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                behaviour=Code.ACTUAL,
                units=Code.BYTES,
                additive=0,
                default=b"",
                fill_value=b"\x00",
            ),
            id_=dict(
                typename="id",
                direction=Code.ANY,
                bytes_expected=1,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                default=b"",
            ),
            operation=dict(
                typename="operation",
                direction=Code.ANY,
                bytes_expected=1,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                descs={0: Code.READ, 1: Code.WRITE},
                default=b"",
            ),
            response=dict(
                typename="response",
                direction=Code.ANY,
                bytes_expected=1,
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                descs={},
                default=b"",
            ),
        )

        for typename, ref in cases.items():
            with self.subTest(classmethod=typename):
                self.assertDictEqual(
                    ref,
                    getattr(
                        TIFieldPattern, typename
                    )().__init_kwargs__(),
                )

    def test_init_exc(self) -> None:
        with self.assertRaises(NotAmongTheOptions) as exc:
            TIFieldPattern("", direction=Code.NONE)
        self.assertEqual(
            "direction option <Code.NONE: 0> not in {<Code.RX: 1554>, "
            "<Code.TX: 1555>, <Code.ANY: 5>}",
            exc.exception.args[0]
        )

    def test_get(self) -> None:
        validate_object(
            self,
            TIFieldPattern(
                typename="crc", default=b"aa", fill_value=b""
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
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_get_specific(self) -> None:
        cases = dict(
            basic=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                default=b"",
                stop=None,
                fill_value=b"",
            ),
            static=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                default=b"\x00",
                stop=1,
                fill_value=b"",
            ),
            address=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                behaviour=Code.DMA,
                units=Code.WORDS,
                default=b"",
                stop=1,
                fill_value=b"",
            ),
            crc=dict(
                fmt=Code.U16,
                order=Code.BIG_ENDIAN,
                poly=0x1021,
                init=0,
                default=b"",
                wo_fields=set(),
                stop=2,
                fill_value=b"\x00",
            ),
            data=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                stop=None,
                default=b"",
                fill_value=b"",
            ),
            dynamic_length=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                behaviour=Code.ACTUAL,
                units=Code.BYTES,
                additive=0,
                default=b"",
                stop=1,
                fill_value=b"\x00",
            ),
            id_=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                default=b"",
                stop=1,
                fill_value=b"",
            ),
            operation=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                descs={0: Code.READ, 1: Code.WRITE},
                descs_r={Code.READ: 0, Code.WRITE: 1},
                default=b"",
                stop=1,
                fill_value=b"",
            ),
            response=dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                descs={},
                descs_r={},
                default=b"",
                stop=1,
                fill_value=b"",
            ),
        )

        for typename, ref in cases.items():
            with self.subTest(classmethod=typename):
                validate_object(
                    self,
                    getattr(
                        TIFieldPattern, typename
                    )().get(),
                    **ref,
                    wo_attrs=[
                        "bytes_expected",
                        "encoder",
                        "name",
                        "start",
                        "is_dynamic",
                        "word_bytesize",
                        "words_expected",
                        "slice_",
                        "has_default",
                        "is_single",
                        "has_fill_value"
                    ],
                )


class TestMessageStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStructPattern(typename="basic").configure(
                f0=TIFieldPattern(
                    typename="static", default=b"a"
                ),
                f1=TIFieldPattern(typename="data"),
            ),
            typename="basic",
            sub_pattern_names=["f0", "f1"],
        )

    def test_init_specific(self) -> None:
        self.assertDictEqual(
            dict(
                typename="basic",
                mtu=1500,
                divisible=False,
            ),
            TIStructPattern.basic().__init_kwargs__()
        )

    def test_get(self) -> None:
        pattern = TIStructPattern.basic(
            divisible=True
        ).configure(
            f0=TIFieldPattern.static(default=b"a"),
            f1=TIFieldPattern.data(),
        )
        msg = pattern.get(
            changes_allowed=True,
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
            slice_=slice(1, None),
            start=1,
            order=Code.BIG_ENDIAN,
            words_expected=0,
            bytes_expected=0,
            fmt=Code.U16,
            is_dynamic=True,
            stop=None,
            is_single=False,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_instance_for_direction(self) -> None:
        ref = TIStructPattern.basic().configure(
                f0=TIFieldPattern.static(),
                f1=TIFieldPattern.response(direction=Code.RX),
                f2=TIFieldPattern.data(direction=Code.TX),
        )

        self.assertListEqual(["f0", "f1", "f2"], ref.sub_pattern_names)
        self.assertIs(ref, ref.instance_for_direction(Code.ANY))
        self.assertListEqual(
            ["f0", "f1"],
            ref.instance_for_direction(Code.RX).sub_pattern_names
        )
        self.assertListEqual(
            ["f0", "f2"],
            ref.instance_for_direction(Code.TX).sub_pattern_names
        )

    def test_instance_for_direction_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIStructPattern.basic().configure(
                f0=TIFieldPattern.static(),
            ).instance_for_direction(Code.NONE)
        self.assertEqual(
            "invalid direction: <Code.NONE: 0>", exc.exception.args[0]
        )


class TestMessagePatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessagePattern.basic().configure(
                s0=TIStructPattern.basic().configure(
                    f0=TIFieldPattern.static(default=b"a"),
                    f1=TIFieldPattern.data(),
                ),
            ),
            typename="basic",
            sub_pattern_names=["s0"],
        )

    def test_get(self) -> None:
        pattern = TIMessagePattern.basic().configure(
            s0=TIStructPattern.basic(
                divisible=True
            ).configure(
                f0=TIFieldPattern.static(default=b"a"),
                f1=TIFieldPattern.data(),
            ),
        )
        msg = pattern.get(
            changes_allowed=True,
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
            slice_=slice(1, None),
            start=1,
            order=Code.BIG_ENDIAN,
            words_expected=0,
            bytes_expected=0,
            fmt=Code.U16,
            is_dynamic=True,
            stop=None,
            is_single=False,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_get_for_direction(self) -> None:
        pattern = TIMessagePattern.basic().configure(
            s0=TIStructPattern.basic().configure(
                f0=TIFieldPattern.static(),
                f1=TIFieldPattern.response(direction=Code.RX),
                f2=TIFieldPattern.data(direction=Code.TX),
            ),
        )

        act = pattern.get_for_direction(Code.ANY)
        self.assertIs(act.pattern, pattern)
        self.assertEqual(
            ["f0", "f1", "f2"], [i[0] for i in act.struct.items()]
        )

        act = pattern.get_for_direction(Code.RX)
        self.assertIs(act.pattern, pattern)
        self.assertEqual(
            ["f0", "f1"], [i[0] for i in act.struct.items()]
        )

        act = pattern.get_for_direction(Code.TX)
        self.assertIs(act.pattern, pattern)
        self.assertEqual(
            ["f0", "f2"], [i[0] for i in act.struct.items()]
        )

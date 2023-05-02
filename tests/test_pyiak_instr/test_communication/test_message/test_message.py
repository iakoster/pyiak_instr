import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.communication.message import (
    Message,
    MessageFieldPattern,
    MessageFieldStruct,
    SingleMessageFieldStruct,
    StaticMessageFieldStruct,
    AddressMessageFieldStruct,
    CrcMessageFieldStruct,
    DataMessageFieldStruct,
    DataLengthMessageFieldStruct,
    IdMessageFieldStruct,
    OperationMessageFieldStruct,
    ResponseMessageFieldStruct,
    MessagePattern
)

from ....utils import validate_object


class TestMessage(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            Message({"f0": MessageFieldStruct()}),
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

    def test_get_has(self) -> None:
        instance = Message(dict(
            f0=SingleMessageFieldStruct(start=0, stop=1),
            f1=AddressMessageFieldStruct(start=1, stop=2),
            f2=OperationMessageFieldStruct(start=2, stop=3),
            f3=DataLengthMessageFieldStruct(start=3, stop=4),
            f4=DataMessageFieldStruct(start=4, stop=-3),
            f5=CrcMessageFieldStruct(start=-3, stop=-1, fmt=Code.U16),
            f6=SingleMessageFieldStruct(start=-1, stop=None),
        ))

        cases = [
            ("basic", False, ""),
            ("single", True, "f0"),
            ("static", False, ""),
            ("address", True, "f1"),
            ("crc", True, "f5"),
            ("data", True, "f4"),
            ("data_length", True, "f3"),
            ("id_", False, ""),
            ("operation", True, "f2"),
            ("response", False, ""),
        ]

        for f_type, has_ref, get_ref in cases:
            with self.subTest(test="has"):
                self.assertEqual(has_ref, getattr(instance.has, f_type))

            with self.subTest(test="get"):
                if has_ref:
                    self.assertEqual(
                        get_ref, getattr(instance.get, f_type).name
                    )
                else:
                    with self.assertRaises(TypeError):
                        getattr(instance.get, f_type)


class TestMessagePattern(unittest.TestCase):

    def test_get(self) -> None:
        res = MessagePattern("basic", "test").configure(
            f0=MessageFieldPattern.basic(fmt=Code.U8, bytes_expected=2),
            f1=MessageFieldPattern.single(fmt=Code.U16),
            f2=MessageFieldPattern.static(fmt=Code.U32, default=b"iak_"),
            f3=MessageFieldPattern.address(fmt=Code.U8),
            f4=MessageFieldPattern.crc(fmt=Code.U16, wo_fields={"f0"}),
            f5=MessageFieldPattern.data(fmt=Code.U8),
            f6=MessageFieldPattern.data_length(fmt=Code.U8, additive=1),
            f7=MessageFieldPattern.id_(fmt=Code.U16),
            f8=MessageFieldPattern.operation(fmt=Code.U8),
            f9=MessageFieldPattern.response(fmt=Code.U8, descs={8: Code.DMA}),
        ).get()

        validate_object(
            self,
            res,
            content=b"",
            divisible=False,
            dst=None,
            is_dynamic=True,
            minimum_size=16,
            mtu=1500,
            name="test",
            src=None,
            src_dst=(None, None),
            wo_attrs=["get", "has"]
        )

        for field, pars in dict(
            f0=dict(
                fmt=Code.U8,
                slice_=slice(0, 2),
                words_expected=2,
            ),
            f1=dict(
                fmt=Code.U16,
                slice_=slice(2, 4),
                words_expected=1,
            ),
            f2=dict(
                fmt=Code.U32,
                slice_=slice(4, 8),
                words_expected=1,
            ),
            f3=dict(
                behaviour=Code.DMA,
                fmt=Code.U8,
                slice_=slice(8, 9),
                words_expected=1,
            ),
            f4=dict(
                fmt=Code.U16,
                init=0,
                poly=0x1021,
                slice_=slice(9, 11),
                wo_fields={"f0"},
                words_expected=1,
            ),
            f5=dict(
                fmt=Code.U8,
                slice_=slice(11, -5),
                words_expected=0,
            ),
            f6=dict(
                additive=1,
                behaviour=Code.ACTUAL,
                fmt=Code.U8,
                slice_=slice(-5, -4),
                units=Code.BYTES,
                words_expected=1,
            ),
            f7=dict(
                fmt=Code.U16,
                slice_=slice(-4, -2),
                words_expected=1,
            ),
            f8=dict(
                descs={0: Code.READ, 1: Code.WRITE},
                descs_r={Code.READ: 0, Code.WRITE: 1},
                fmt=Code.U8,
                slice_=slice(-2, -1),
                words_expected=1,
            ),
            f9=dict(
                descs={8: Code.DMA},
                descs_r={Code.DMA: 8},
                fmt=Code.U8,
                slice_=slice(-1, None),
                words_expected=1,
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

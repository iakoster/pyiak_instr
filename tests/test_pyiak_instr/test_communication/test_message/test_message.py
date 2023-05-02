import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.communication.message import (
    Message,
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

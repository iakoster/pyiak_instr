import unittest

from src.pyiak_instr.core import Code

from .....utils import validate_object, get_object_attrs
from .ti import (
    TIMessageFieldStruct,
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
)


class TestMessageABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessage(TIMessageStruct(fields=dict(
                f0=TIDataLengthMessageFieldStruct(
                    name="f0", stop=2, fmt=Code.U16
                ),
                f1=TIDataMessageFieldStruct(name="f1", start=2, fmt=Code.U32),
            ))),
            has_pattern=False,
            src=None,
            dst=None,
            wo_attrs=["struct", "get", "has"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIMessage(TIMessageStruct(
                fields={
                    "f0": TIAddressMessageFieldStruct(
                        name="f0", behaviour=Code.STRONG
                    ),
                    "f1": TIDataMessageFieldStruct(name="f1"),
                },
                divisible=True,
            ))
        self.assertEqual(
            "invalid address behaviour for divisible message: "
            "<Code.STRONG: 1540>",
            exc.exception.args[0],
        )

    def test_autoupdate_fields(self) -> None:
        with self.subTest(test="basic"):
            obj = TIMessage(TIMessageStruct(fields=dict(
                data_length=TIDataLengthMessageFieldStruct(
                    name="data_length", stop=2, fmt=Code.U16, additive=1
                ),
                data=TIDataMessageFieldStruct(
                    name="data", start=2, stop=-2, fmt=Code.U16
                ),
                crc=TICrcMessageFieldStruct(
                    name="crc", start=-2, fmt=Code.U16
                ),
            )))
            obj.encode(bytes(range(10)))
            obj.autoupdate_fields()

            self.assertEqual(
                b"\x00\x07\x02\x03\x04\x05\x06\x07\x87\x96", obj.content()
            )

        with self.subTest(test="dynamic length is expected"):
            obj = TIMessage(TIMessageStruct(fields=dict(
                data_length=TIDataLengthMessageFieldStruct(
                    name="data_length",
                    stop=2,
                    fmt=Code.U16,
                    behaviour=Code.EXPECTED,
                ),
                data=TIDataMessageFieldStruct(
                    name="data", start=2, fmt=Code.U16
                ),
            ))).encode(bytes(range(6))).autoupdate_fields()

            self.assertEqual(bytes(range(6)), obj.content())

    def test_autoupdate_fields_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIMessage(TIMessageStruct(fields=dict(
                data=TIDataMessageFieldStruct(name="data"),
            ))).autoupdate_fields()
        self.assertEqual("message is empty", exc.exception.args[0])

    def test_has(self) -> None:
        obj = TIMessage(TIMessageStruct(fields=dict(
            f0=TIMessageFieldStruct(name="f0", stop=1),
            f2=TIStaticMessageFieldStruct(name="f2", start=2, stop=3, default=b"a"),
            f3=TIAddressMessageFieldStruct(name="f3", start=3, stop=4),
            f4=TICrcMessageFieldStruct(name="f4", start=4, stop=6, fmt=Code.U16),
            f5=TIDataMessageFieldStruct(name="f5", start=6, stop=-4),
            f6=TIDataLengthMessageFieldStruct(name="f6", start=-4, stop=-3),
            f7=TIIdMessageFieldStruct(name="f7", start=-3, stop=-2),
            f8=TIOperationMessageFieldStruct(name="f8", start=-2),
        )))

        validate_object(
            self,
            obj.has,
            basic=True,
            address=True,
            id_=True,
            data_length=True,
            response=False,
            static=True,
            crc=True,
            operation=True,
            data=True,
        )

        self.assertFalse(obj.has(Code.UNDEFINED))

    def test_get(self) -> None:
        obj = TIMessage(TIMessageStruct(fields=dict(
            f0=TIMessageFieldStruct(name="f0", stop=1),
            f2=TIStaticMessageFieldStruct(name="f2", start=2, stop=3, default=b"a"),
            f3=TIAddressMessageFieldStruct(name="f3", start=3, stop=4),
            f4=TICrcMessageFieldStruct(name="f4", start=4, stop=6, fmt=Code.U16),
            f5=TIDataMessageFieldStruct(name="f5", start=6, stop=-4),
            f6=TIDataLengthMessageFieldStruct(name="f6", start=-4, stop=-3),
            f7=TIIdMessageFieldStruct(name="f7", start=-3, stop=-2),
            f8=TIOperationMessageFieldStruct(name="f8", start=-2, stop=-1),
            f9=TIResponseMessageFieldStruct(name="f9", start=-1)
        )))

        ref = dict(
            basic="f0",
            address="f3",
            id_="f7",
            data_length="f6",
            response="f9",
            static="f2",
            crc="f4",
            operation="f8",
            data="f5",
        )
        get = obj.get
        for attr in get_object_attrs(get):
            with self.subTest(field=attr):
                self.assertEqual(ref[attr], getattr(get, attr).name)

    def test_src_dst(self) -> None:
        obj = TIMessage(TIMessageStruct(fields={
            "f0": TIDataLengthMessageFieldStruct(name="f0"),
        }))

        self.assertTupleEqual((None, None), (obj.src, obj.dst))
        obj.src = "123"
        obj.dst = 456
        self.assertTupleEqual(("123", 456), (obj.src, obj.dst))

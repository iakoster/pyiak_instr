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

    def test_split(self) -> None:
        # with self.subTest(test="short message"):
        #     obj = self._instance().encode(bytes(range(9))).autoupdate_fields()
        #     for msg in obj.split():
        #         self.assertIs(obj, msg)

        with self.subTest(test="even long message"):
            obj = self._instance().encode(
                f0=0x23,
                f1=1,
                f2=4,
                f3=0,
                f4=[*range(24)],
                f5=0,
                f6=0x33,
                f7=0x11,
                f8=0x32,
            ).autoupdate_fields()

            parts = [
                bytes([35, 1, 4, 218, 41, *range(12), 12, 51, 17, 50]),
                bytes([35, 1, 16, 173, 112, *range(12, 24), 12, 51, 17, 50]),
            ]
            for i, msg in enumerate(obj.split()):
                with self.subTest(message=i):
                    self.assertEqual(parts[i].hex(" "), msg.content().hex(" "))

        with self.subTest(test="non-even long message"):
            obj = self._instance(
                address_units=Code.WORDS,
                data_fmt=Code.U16,
            ).encode(
                f0=0,
                f1=1,
                f2=0,
                f3=0,
                f4=[*range(10)],
                f5=0,
                f6=0,
                f7=0,
                f8=0,
            ).autoupdate_fields()

            parts = [
                bytes([0, 1, 0, 149, 6, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 12, 0, 0, 0]),
                bytes([0, 1, 6, 10, 71, 0, 6, 0, 7, 0, 8, 0, 9, 8, 0, 0, 0]),
            ]
            for i, msg in enumerate(obj.split()):
                with self.subTest(message=i):
                    self.assertEqual(parts[i].hex(" "), msg.content().hex(" "))

    def test_split_exc(self) -> None:
        with self.subTest(test="empty field"):
            with self.assertRaises(ValueError) as exc:
                next(self._instance().split())
            self.assertEqual("message is empty", exc.exception.args[0])

    def test_has(self) -> None:
        obj = self._instance()

        validate_object(
            self,
            obj.has,
            basic=True,
            address=True,
            id_=True,
            data_length=True,
            response=True,
            static=True,
            crc=True,
            operation=True,
            data=True,
        )

        self.assertFalse(obj.has(Code.UNDEFINED))

    def test_get(self) -> None:
        obj = self._instance()

        ref = dict(
            basic="f0",
            address="f2",
            id_="f6",
            data_length="f5",
            response="f8",
            static="f1",
            crc="f3",
            operation="f7",
            data="f4",
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

    @staticmethod
    def _instance(
            address_units: Code = Code.BYTES,
            data_fmt=Code.U8,
    ) -> TIMessage:
        return TIMessage(TIMessageStruct(
            fields=dict(
                f0=TIMessageFieldStruct(name="f0", stop=1),
                f1=TIStaticMessageFieldStruct(name="f1", start=1, stop=2, default=b"\x01"),
                f2=TIAddressMessageFieldStruct(
                    name="f2", start=2, stop=3, units=address_units
                ),
                f3=TICrcMessageFieldStruct(name="f3", start=3, stop=5, fmt=Code.U16),
                f4=TIDataMessageFieldStruct(name="f4", start=5, stop=-4, fmt=data_fmt),
                f5=TIDataLengthMessageFieldStruct(name="f5", start=-4, stop=-3),
                f6=TIIdMessageFieldStruct(name="f6", start=-3, stop=-2),
                f7=TIOperationMessageFieldStruct(name="f7", start=-2, stop=-1),
                f8=TIResponseMessageFieldStruct(name="f8", start=-1)
            ),
            divisible=True,
            mtu=21,
        ))

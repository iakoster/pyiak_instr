import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.encoders import BytesEncoder
from src.pyiak_instr.exceptions import NotAmongTheOptions, ContentError
from src.pyiak_instr.communication.message.types import MessageStructABC

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
)


class TestMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessageFieldStruct(
                start=10, fmt=Code.I64, bytes_expected=160
            ),
            bytes_expected=160,
            default=b"",
            fmt=Code.I64,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(10, 170),
            start=10,
            stop=170,
            word_bytesize=8,
            words_expected=20,
            name="",
            is_single=False,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIAddressMessageFieldStruct(start=0, fmt=Code.U8, stop=2)
        self.assertEqual(
            "TIAddressMessageFieldStruct should expect one word",
            exc.exception.args[0],
        )


class TestStaticMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStaticMessageFieldStruct(start=0, fmt=Code.U8, default=b"\x00"),
            bytes_expected=1,
            default=b"\x00",
            fmt=Code.U8,
            has_default=True,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            word_bytesize=1,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIStaticMessageFieldStruct(start=0, fmt=Code.U8)
        self.assertEqual("default value not specified", exc.exception.args[0])

    def test_verify(self) -> None:
        for i, (data, ref) in enumerate(((b"2", False), (b"4", True))):
            with self.subTest(test=i):
                self.assertEqual(ref, TIStaticMessageFieldStruct(
                    start=0, fmt=Code.U8, default=b"4"
                ).verify(data))

    def test_verify_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            TIStaticMessageFieldStruct(
                default=b"a"
            ).verify(b"b", raise_if_false=True)
        self.assertEqual(
            "invalid content in TIStaticMessageFieldStruct: 62",
            exc.exception.args[0],
        )


class TestAddressMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIAddressMessageFieldStruct(start=0, fmt=Code.U8),
            behaviour=Code.DMA,
            bytes_expected=1,
            default=b"",
            fmt=Code.U8,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 1),
            start=0,
            stop=1,
            units=Code.WORDS,
            word_bytesize=1,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid behaviour"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIAddressMessageFieldStruct(
                    start=0, fmt=Code.U8, behaviour=Code.EXPECTED
                )
            self.assertEqual(
                "behaviour option not in {<Code.DMA: 1539>, "
                "<Code.STRONG: 1540>}, got <Code.EXPECTED: 1541>",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid units"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIAddressMessageFieldStruct(units=Code.NONE)
            self.assertEqual(
                "units option not in {<Code.WORDS: 768>, "
                "<Code.BYTES: 257>}, got <Code.NONE: 0>",
                exc.exception.args[0]
            )


class TestCrcMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TICrcMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            init=0,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            poly=0x1021,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            wo_fields=set(),
            word_bytesize=2,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"\x00",
            has_fill_value=True,
            wo_attrs=["encoder"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(NotImplementedError) as exc:
            TICrcMessageFieldStruct(poly=0x1020)
        self.assertEqual(
            "Crc algorithm not verified for other values",
            exc.exception.args[0],
        )

    def test_calculate(self) -> None:
        contents = (
            b"\xaa",
            b"\xaa\xbb",
            b"\xff\xaa\x88\x44",
            b"\xff\xaa\x88\x44\xff\xaa\x88\x44",
            b"\x00\x10\x20\x30\x40\x50\x60\x70\x80\x90\xa0\xb0\xc0\xd0\xe0"
            b"\xf0",
        )
        data = {
            "CRC-16/XMODEM": dict(
                fmt=Code.U16,
                init=0,
                poly=0x1021,
                crc=(0x14A0, 0xE405, 0xAB8F, 0x488D, 0x4375),
            )
        }
        for name in data:
            dict_: dict = data[name].copy()
            crcs = dict_.pop("crc")
            obj = TICrcMessageFieldStruct(start=0, **dict_)

            assert len(contents) == len(crcs)
            for i, (content, crc) in enumerate(zip(contents, crcs)):
                with self.subTest(test=name, crc=i):
                    self.assertEqual(crc, obj.calculate(content))


class TestDataMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIDataMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=0,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            is_dynamic=True,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, None),
            start=0,
            stop=None,
            word_bytesize=2,
            words_expected=0,
            name="",
            is_single=False,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="data not dynamic"):
            with self.assertRaises(ValueError) as exc:
                TIDataMessageFieldStruct(stop=1)
            self.assertEqual(
                "TIDataMessageFieldStruct can only be dynamic",
                exc.exception.args[0],
            )


class TestDataLengthMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIDataLengthMessageFieldStruct(start=0, fmt=Code.U16),
            additive=0,
            behaviour=Code.ACTUAL,
            bytes_expected=2,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            units=Code.BYTES,
            word_bytesize=2,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"\x00",
            has_fill_value=True,
            wo_attrs=["encoder"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="negative additive"):
            with self.assertRaises(ValueError) as exc:
                TIDataLengthMessageFieldStruct(
                    start=0, fmt=Code.U8, additive=-1
                )
            self.assertEqual(
                "additive number must be positive integer",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid behaviour"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIDataLengthMessageFieldStruct(
                    start=0, fmt=Code.U8, behaviour=Code.DMA
                )
            self.assertEqual(
                "behaviour option not allowed, got <Code.DMA: 1539>",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid units"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIDataLengthMessageFieldStruct(
                    start=0, fmt=Code.U8, units=Code.INT
                )
            self.assertEqual(
                "units option not allowed, got <Code.INT: 261>",
                exc.exception.args[0],
            )

    def test_calculate(self) -> None:
        with self.subTest(test="U8 BYTES"):
            self.assertEqual(
                20,
                TIDataLengthMessageFieldStruct().calculate(
                    b"a" * 20, BytesEncoder(fmt=Code.U8).value_size,
                ),
            )

        with self.subTest(test="U16 BYTES"):
            self.assertEqual(
                20,
                TIDataLengthMessageFieldStruct(
                    units=Code.BYTES
                ).calculate(b"a" * 20, BytesEncoder(fmt=Code.U16).value_size)
            )

        with self.subTest(test="U32 WORDS"):
            self.assertEqual(
                4,
                TIDataLengthMessageFieldStruct(
                    units=Code.WORDS
                ).calculate(b"a" * 16, BytesEncoder(fmt=Code.U32).value_size)
            )

    def test_calculate_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            TIDataLengthMessageFieldStruct(
                units=Code.WORDS
            ).calculate(b"a" * 5, BytesEncoder(fmt=Code.U32).value_size)
        self.assertEqual(
            "invalid content in TIDataLengthMessageFieldStruct: "
            "non-integer words count in data",
            exc.exception.args[0],
        )


class TestIdMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIIdMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            word_bytesize=2,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )


class TestOperationMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIOperationMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            descs={0: Code.READ, 1: Code.WRITE},
            descs_r={Code.READ: 0, Code.WRITE: 1},
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            word_bytesize=2,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_encode(self) -> None:
        self.assertEqual(
            b"\x00",
            TIOperationMessageFieldStruct().encode(Code.READ)
        )

    def test_encode_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            TIOperationMessageFieldStruct().encode(Code.UNDEFINED)
        self.assertEqual(
            "invalid content in TIOperationMessageFieldStruct: "
            "can't encode <Code.UNDEFINED: 255>",
            exc.exception.msg,
        )

    def test_desc(self) -> None:
        obj = TIOperationMessageFieldStruct()
        cases = (
            (0, Code.READ),
            (1, Code.WRITE),
            (2, Code.UNDEFINED),
        )
        for input_, ref in cases:
            with self.subTest(ref=repr(ref)):
                self.assertEqual(ref, obj.desc(input_))

    def test_desc_r(self) -> None:
        obj = TIOperationMessageFieldStruct()
        cases = (
            (Code.READ, 0),
            (Code.WRITE, 1),
            (Code.UNDEFINED, None),
        )
        for input_, ref in cases:
            with self.subTest(ref=repr(ref)):
                self.assertEqual(ref, obj.desc_r(input_))


class TestResponseMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIResponseMessageFieldStruct(start=0, fmt=Code.U16),
            bytes_expected=2,
            default=b"",
            descs={},
            descs_r={},
            fmt=Code.U16,
            has_default=False,
            is_dynamic=False,
            order=Code.BIG_ENDIAN,
            slice_=slice(0, 2),
            start=0,
            stop=2,
            word_bytesize=2,
            words_expected=1,
            name="",
            is_single=True,
            fill_value=b"",
            has_fill_value=False,
            wo_attrs=["encoder"],
        )

    def test_encode(self) -> None:
        self.assertEqual(
            b"\x00",
            self._instance.encode(Code.U8)
        )

    def test_encode_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            self._instance.encode(Code.UNDEFINED)
        self.assertEqual(
            "invalid content in TIResponseMessageFieldStruct: "
            "can't encode <Code.UNDEFINED: 255>",
            exc.exception.msg,
        )

    def test_desc(self) -> None:
        cases = (
            (0, Code.U8),
            (1, Code.WAIT),
            (2, Code.UNDEFINED),
        )
        for input_, ref in cases:
            with self.subTest(ref=repr(ref)):
                self.assertEqual(ref, self._instance.desc(input_))

    def test_desc_r(self) -> None:
        cases = (
            (Code.U8, 0),
            (Code.WAIT, 1),
            (Code.UNDEFINED, None),
        )
        for input_, ref in cases:
            with self.subTest(ref=repr(ref)):
                self.assertEqual(ref, self._instance.desc_r(input_))

    @property
    def _instance(self) -> TIResponseMessageFieldStruct:
        return TIResponseMessageFieldStruct(descs={0: Code.U8, 1: Code.WAIT})


class TestMessageStructABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIMessageStruct(fields={"f": TIMessageFieldStruct(name="f")}),
            divisible=False,
            dynamic_field_name="f",
            minimum_size=0,
            is_dynamic=True,
            mtu=1500,
            name="std",
            wo_attrs=["fields", "get", "has"]
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="divisible without dynamic"):
            with self.assertRaises(TypeError) as exc:
                TIMessageStruct(
                    fields={"f": TIMessageFieldStruct(name="f", stop=1)},
                    divisible=True,
                )
            self.assertEqual(
                "TIMessageStruct can not be divided because it does not "
                "have a dynamic field",
                exc.exception.args[0],
            )

        with self.subTest(test="small mtu"):
            with self.assertRaises(ValueError) as exc:
                TIMessageStruct(
                    fields={
                        "f0": TIMessageFieldStruct(name="f0", stop=4),
                        "f1": TIMessageFieldStruct(name="f1", start=4),
                    },
                    divisible=True,
                    mtu=4,
                )
            self.assertEqual(
                "MTU value does not allow you to split the message if "
                "necessary. The minimum MTU is 5 (got 4)",
                exc.exception.args[0],
            )

        with self.subTest(test="not specified code"):
            with self.assertRaises(KeyError) as exc:
                MessageStructABC(fields={
                    "f0": TIMessageFieldStruct(name="f0")
                })
            self.assertEqual(
                "TIMessageFieldStruct not represented in codes",
                exc.exception.args[0],
            )

    def test_has(self) -> None:
        obj = TIMessageStruct(fields=dict(
            f0=TIMessageFieldStruct(name="f0", stop=1),
            f2=TIStaticMessageFieldStruct(name="f2", start=2, stop=3, default=b"a"),
            f3=TIAddressMessageFieldStruct(name="f3", start=3, stop=4),
            f4=TICrcMessageFieldStruct(name="f4", start=4, stop=6, fmt=Code.U16),
            f5=TIDataMessageFieldStruct(name="f5", start=6, stop=-4),
            f6=TIDataLengthMessageFieldStruct(name="f6", start=-4, stop=-3),
            f7=TIIdMessageFieldStruct(name="f7", start=-3, stop=-2),
            f8=TIOperationMessageFieldStruct(name="f8", start=-2),
        ))

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
        obj = TIMessageStruct(fields=dict(
            f0=TIMessageFieldStruct(name="f0", stop=1),
            f2=TIStaticMessageFieldStruct(name="f2", start=2, stop=3, default=b"a"),
            f3=TIAddressMessageFieldStruct(name="f3", start=3, stop=4),
            f4=TICrcMessageFieldStruct(name="f4", start=4, stop=6, fmt=Code.U16),
            f5=TIDataMessageFieldStruct(name="f5", start=6, stop=-4),
            f6=TIDataLengthMessageFieldStruct(name="f6", start=-4, stop=-3),
            f7=TIIdMessageFieldStruct(name="f7", start=-3, stop=-2),
            f8=TIOperationMessageFieldStruct(name="f8", start=-2, stop=-1),
            f9=TIResponseMessageFieldStruct(name="f9", start=-1)
        ))

        ref = dict(
            basic="f0",
            single="f1",
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

    def test_get_exc(self) -> None:
        obj = TIMessageStruct(fields=dict(
            f0=TIMessageFieldStruct(name="f0", stop=1),
        ))

        with self.subTest(test="invalid code"):
            with self.assertRaises(TypeError) as exc:
                obj.get(Code.UNDEFINED)
            self.assertEqual(
                "field instance with code <Code.UNDEFINED: 255> not found",
                exc.exception.args[0],
            )

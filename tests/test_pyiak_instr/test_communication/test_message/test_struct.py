import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.codecs import get_bytes_codec
from src.pyiak_instr.exceptions import NotAmongTheOptions, ContentError

from tests.utils import validate_object, get_object_attrs

from src.pyiak_instr.communication.message import STRUCT_DATACLASS
from tests.pyiak_instr_ti.communication.message import (
    TIBasic,
    TIStatic,
    TIAddress,
    TICrc,
    TIData,
    TIDynamicLength,
    TIId,
    TIOperation,
    TIResponse,
    TIStruct,
)


@STRUCT_DATACLASS
class TIBasicAnother(TIBasic):
    ...


class TestMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIBasic(
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
            wo_attrs=["codec"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIAddress(start=0, fmt=Code.U8, stop=2)
        self.assertEqual(
            "TIAddress should expect one word",
            exc.exception.args[0],
        )


class TestStaticMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStatic(start=0, fmt=Code.U8, default=b"\x00"),
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
            wo_attrs=["codec"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIStatic(start=0, fmt=Code.U8)
        self.assertEqual("default value not specified", exc.exception.args[0])

    def test_verify(self) -> None:
        for i, (data, ref) in enumerate(
                ((b"2", Code.INVALID_CONTENT), (b"4", Code.OK))
        ):
            with self.subTest(test=i):
                self.assertIs(ref, TIStatic(
                    start=0, fmt=Code.U8, default=b"4"
                ).verify(data))

    def test_verify_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            TIStatic(
                default=b"a"
            ).verify(b"b", raise_if_false=True)
        self.assertEqual(
            "invalid content in TIStatic: <Code.INVALID_CONTENT: 1028> - '62'",
            exc.exception.args[0],
        )


class TestAddressMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIAddress(start=0, fmt=Code.U8),
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
            wo_attrs=["codec"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid behaviour"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIAddress(
                    start=0, fmt=Code.U8, behaviour=Code.EXPECTED
                )
            self.assertEqual(
                "'behaviour' option <Code.EXPECTED: 1541> not in "
                "{<Code.DMA: 1539>, <Code.STRONG: 1540>}",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid units"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIAddress(units=Code.NONE)
            self.assertEqual(
                "'units' option <Code.NONE: 0> not in {<Code.WORDS: 768>, "
                "<Code.BYTES: 257>}",
                exc.exception.args[0]
            )


class TestCrcMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TICrc(start=0, fmt=Code.U16),
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
            fill_content=b"\x00\x00",
            has_fill_value=True,
            wo_attrs=["codec"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(NotImplementedError) as exc:
            TICrc(poly=0x1020)
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
            obj = TICrc(start=0, **dict_)

            assert len(contents) == len(crcs)
            for i, (content, crc) in enumerate(zip(contents, crcs)):
                with self.subTest(test=name, crc=i):
                    self.assertEqual(crc, obj.calculate(content))


class TestDataMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIData(start=0, fmt=Code.U16),
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
            wo_attrs=["codec"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="data not dynamic"):
            with self.assertRaises(ValueError) as exc:
                TIData(stop=1)
            self.assertEqual(
                "TIData can only be dynamic",
                exc.exception.args[0],
            )


class TestDataLengthMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIDynamicLength(start=0, fmt=Code.U16),
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
            fill_content=b"\x00\x00",
            has_fill_value=True,
            wo_attrs=["codec"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="negative additive"):
            with self.assertRaises(ValueError) as exc:
                TIDynamicLength(
                    start=0, fmt=Code.U8, additive=-1
                )
            self.assertEqual(
                "additive number must be positive integer",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid behaviour"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIDynamicLength(
                    start=0, fmt=Code.U8, behaviour=Code.DMA
                )
            self.assertEqual(
                "'behaviour' option <Code.DMA: 1539> not allowed",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid units"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIDynamicLength(
                    start=0, fmt=Code.U8, units=Code.INT
                )
            self.assertEqual(
                "'units' option <Code.INT: 261> not allowed",
                exc.exception.args[0],
            )

    def test_calculate(self) -> None:
        with self.subTest(test="U8 BYTES"):
            self.assertEqual(
                20,
                TIDynamicLength().calculate(
                    b"a" * 20, get_bytes_codec(fmt=Code.U8).fmt_bytesize,
                ),
            )

        with self.subTest(test="U16 BYTES"):
            self.assertEqual(
                20,
                TIDynamicLength(
                    units=Code.BYTES
                ).calculate(b"a" * 20, get_bytes_codec(fmt=Code.U16).fmt_bytesize)
            )

        with self.subTest(test="U32 WORDS"):
            self.assertEqual(
                4,
                TIDynamicLength(
                    units=Code.WORDS
                ).calculate(b"a" * 16, get_bytes_codec(fmt=Code.U32).fmt_bytesize)
            )

    def test_calculate_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            TIDynamicLength(
                units=Code.WORDS
            ).calculate(b"a" * 5, get_bytes_codec(fmt=Code.U32).fmt_bytesize)
        self.assertEqual(
            "invalid content in TIDynamicLength: "
            "non-integer words count in data",
            exc.exception.args[0],
        )


class TestIdMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIId(start=0, fmt=Code.U16),
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
            wo_attrs=["codec"],
        )


class TestOperationMessageFieldStruct(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIOperation(start=0, fmt=Code.U16),
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
            wo_attrs=["codec"],
        )

    def test_encode(self) -> None:
        self.assertEqual(
            b"\x00",
            TIOperation().encode(Code.READ)
        )

    def test_encode_exc(self) -> None:
        with self.assertRaises(ContentError) as exc:
            TIOperation().encode(Code.UNDEFINED)
        self.assertEqual(
            "invalid content in TIOperation: "
            "can't encode <Code.UNDEFINED: 255>",
            exc.exception.msg,
        )

    def test_desc(self) -> None:
        obj = TIOperation()
        cases = (
            (0, Code.READ),
            (1, Code.WRITE),
            (2, Code.UNDEFINED),
        )
        for input_, ref in cases:
            with self.subTest(ref=repr(ref)):
                self.assertEqual(ref, obj.desc(input_))

    def test_desc_r(self) -> None:
        obj = TIOperation()
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
            TIResponse(start=0, fmt=Code.U16),
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
            wo_attrs=["codec"],
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
            "invalid content in TIResponse: "
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
    def _instance(self) -> TIResponse:
        return TIResponse(descs={0: Code.U8, 1: Code.WAIT})


class TestMessageStructABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStruct(fields={"f": TIBasic(name="f")}),
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
                TIStruct(
                    fields={"f": TIBasic(name="f", stop=1)},
                    divisible=True,
                )
            self.assertEqual(
                "TIStruct can not be divided because it does not "
                "have a dynamic field",
                exc.exception.args[0],
            )

        with self.subTest(test="dynamic length field without dynamic field"):
            with self.assertRaises(TypeError) as exc:
                TIStruct(fields=dict(
                    f0=TIDynamicLength(name="f0")
                ))
            self.assertEqual(
                "dynamic length field without dynamic length detected",
                exc.exception.args[0],
            )

        with self.subTest(test="small mtu"):
            with self.assertRaises(ValueError) as exc:
                TIStruct(
                    fields={
                        "f0": TIBasic(name="f0", stop=4),
                        "f1": TIBasic(name="f1", start=4),
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
                TIStruct(fields={
                    "f0": TIBasicAnother(name="f0")
                })
            self.assertEqual(
                "TIBasicAnother not represented in codes",
                exc.exception.args[0],
            )

    def test_has(self) -> None:
        obj = TIStruct(fields=dict(
            f0=TIBasic(name="f0", stop=1),
            f2=TIStatic(name="f2", start=2, stop=3, default=b"a"),
            f3=TIAddress(name="f3", start=3, stop=4),
            f4=TICrc(name="f4", start=4, stop=6, fmt=Code.U16),
            f5=TIData(name="f5", start=6, stop=-4),
            f6=TIDynamicLength(name="f6", start=-4, stop=-3),
            f7=TIId(name="f7", start=-3, stop=-2),
            f8=TIOperation(name="f8", start=-2),
        ))

        validate_object(
            self,
            obj.has,
            basic=True,
            address=True,
            id_=True,
            dynamic_length=True,
            response=False,
            static=True,
            crc=True,
            operation=True,
            data=True,
        )

        self.assertFalse(obj.has(Code.UNDEFINED))

    def test_get(self) -> None:
        obj = TIStruct(fields=dict(
            f0=TIBasic(name="f0", stop=1),
            f2=TIStatic(name="f2", start=2, stop=3, default=b"a"),
            f3=TIAddress(name="f3", start=3, stop=4),
            f4=TICrc(name="f4", start=4, stop=6, fmt=Code.U16),
            f5=TIData(name="f5", start=6, stop=-4),
            f6=TIDynamicLength(name="f6", start=-4, stop=-3),
            f7=TIId(name="f7", start=-3, stop=-2),
            f8=TIOperation(name="f8", start=-2, stop=-1),
            f9=TIResponse(name="f9", start=-1)
        ))

        ref = dict(
            basic="f0",
            single="f1",
            address="f3",
            id_="f7",
            dynamic_length="f6",
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
        obj = TIStruct(fields=dict(
            f0=TIBasic(name="f0", stop=1),
        ))

        with self.subTest(test="invalid code"):
            with self.assertRaises(TypeError) as exc:
                obj.get(Code.UNDEFINED)
            self.assertEqual(
                "field instance with code <Code.UNDEFINED: 255> not found",
                exc.exception.args[0],
            )

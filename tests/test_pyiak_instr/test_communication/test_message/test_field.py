import unittest

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import ContentError
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
    MessageFieldPattern,
)

from ....utils import validate_object


def _get_instance(
        encode: bool = False,
) -> Message:
    instance = Message(
        dict(
            f0=MessageFieldStruct(start=0, stop=2),
            f1=SingleMessageFieldStruct(start=2, stop=4, fmt=Code.U16),
            f2=StaticMessageFieldStruct(
                start=4, stop=8, fmt=Code.U32, default=b"iak_"
            ),
            f3=AddressMessageFieldStruct(start=8, stop=9),
            f4=CrcMessageFieldStruct(
                start=9, stop=11, fmt=Code.U16, wo_fields=["f0"]
            ),
            f5=DataMessageFieldStruct(start=11, stop=-5),
            f6=DataLengthMessageFieldStruct(start=-5, stop=-4, additive=1),
            f7=IdMessageFieldStruct(start=-4, stop=-2, fmt=Code.U16),
            f8=OperationMessageFieldStruct(
                start=-2, stop=-1, descs={7: Code.ERROR}
            ),
            f9=ResponseMessageFieldStruct(
                start=-1, stop=None, descs={8: Code.DMA}
            ),
        ),
        name="test",
    )

    if encode:
        instance.encode(
            f0=[0, 1],
            f1=2,
            f3=3,
            f4=4,
            f6=5,
            f7=6,
            f8=7,
            f9=8
        )

    return instance


class TestMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.basic,
            bytes_count=2,
            content=b"\x00\x01",
            is_empty=False,
            name="f0",
            words_count=2,
            wo_attrs=["struct"]
        )


class TestSingleMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.single,
            bytes_count=2,
            content=b"\x00\x02",
            is_empty=False,
            name="f1",
            words_count=1,
            wo_attrs=["struct"]
        )


class TestStaticMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.static,
            bytes_count=4,
            content=b"iak_",
            is_empty=False,
            name="f2",
            words_count=1,
            wo_attrs=["struct"]
        )


class TestAddressMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.address,
            bytes_count=1,
            content=b"\x03",
            is_empty=False,
            name="f3",
            words_count=1,
            wo_attrs=["struct"]
        )


class TestCrcMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.crc,
            bytes_count=2,
            content=b"\x00\x04",
            is_empty=False,
            name="f4",
            words_count=1,
            wo_attrs=["struct"]
        )

    def test_calculate(self) -> None:
        self.assertEqual(
            0x7441, _get_instance(True).get.crc.calculate()
        )

    def test_update(self) -> None:
        obj = _get_instance(True).get.crc
        self.assertEqual(4, obj[0])
        obj.update()
        self.assertEqual(0x7441, obj[0])

    def test_verify_content(self) -> None:
        with self.subTest(test="empty"):
            with self.assertRaises(ContentError) as exc:
                _get_instance().get.crc.verify_content()
            self.assertEqual(
                "invalid content in CrcMessageField: field is empty",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid crc"):
            with self.assertRaises(ContentError) as exc:
                _get_instance(True).get.crc.verify_content()
            self.assertEqual(
                "invalid content in CrcMessageField: invalid crc value",
                exc.exception.args[0]
            )


class TestDataMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.data,
            bytes_count=0,
            content=b"",
            is_empty=True,
            name="f5",
            words_count=0,
            wo_attrs=["struct"]
        )

    def test_append(self) -> None:
        obj = _get_instance(True).get.data
        self.assertEqual(b"", obj.content)
        obj.append(b"\x00\x02")
        self.assertEqual(b"\x00\x02", obj.content)
        obj.append(b"\x01")
        self.assertEqual(b"\x00\x02\x01", obj.content)

    def test_append_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            Message(
                {"d": DataMessageFieldStruct(stop=1)}
            ).get.data.append(b"")
        self.assertEqual(
            "fails to add new data to a non-dynamic field",
            exc.exception.args[0],
        )


class TestDataLengthMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.data_length,
            bytes_count=1,
            content=b"\x05",
            is_empty=False,
            name="f6",
            words_count=1,
            wo_attrs=["struct"]
        )

    def test_calculate(self) -> None:
        with self.subTest(test="actual behaviour"):
            msg = Message(dict(
                dl=DataLengthMessageFieldStruct(stop=1),
                d=DataMessageFieldStruct(start=1),
            )).encode(dl=0)
            data = msg.get.data
            dl_ = msg.get.data_length
            dl_.verify_content()

            data.append(b"\x00")
            self.assertEqual(1, dl_.calculate())
            data.append(b"a" * 30)
            self.assertEqual(31, dl_.calculate())

        with self.subTest(test="expected behaviour"):
            msg = Message(dict(
                dl=DataLengthMessageFieldStruct(
                    stop=1, behaviour=Code.EXPECTED
                ),
                d=DataMessageFieldStruct(start=1),
            )).encode(dl=0)
            data = msg.get.data
            dl_ = msg.get.data_length
            dl_.verify_content()

            data.append(b"\x00")
            self.assertEqual(1, dl_.calculate())
            dl_.update()
            data.append(b"a" * 30)
            self.assertEqual(1, dl_.calculate())

    def test_update(self) -> None:
        instance = _get_instance(True)
        field = instance.get.data_length
        self.assertEqual(5, field[0])
        field.update()
        self.assertEqual(0, field[0])
        instance.change("f5", [10, 2, 1])
        field.update()
        self.assertEqual(3, field[0])

    def test_verify_content(self) -> None:
        with self.subTest(test="empty"):
            with self.assertRaises(ContentError) as exc:
                _get_instance().get.data_length.verify_content()
            self.assertEqual(
                "invalid content in DataLengthMessageField: field is empty",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid data length"):
            with self.assertRaises(ContentError) as exc:
                _get_instance(True).get.data_length.verify_content()
            self.assertEqual(
                "invalid content in DataLengthMessageField: "
                "invalid data length value",
                exc.exception.args[0]
            )


class TestIdMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.id_,
            bytes_count=2,
            content=b"\x00\x06",
            is_empty=False,
            name="f7",
            words_count=1,
            wo_attrs=["struct"]
        )

    def test_is_equal_to(self) -> None:
        with self.subTest(test="empty field"):
            self.assertFalse(_get_instance().get.id_.is_equal_to(b""))

        msg = _get_instance(True)
        field = msg.get.id_
        with self.subTest(test="id field"):
            self.assertTrue(field.is_equal_to(
                Message(dict(
                    f0=IdMessageFieldStruct(fmt=Code.U16)
                )).encode(f0=6).get.id_
            ))

        with self.subTest(test="list"):
            self.assertTrue(field.is_equal_to([6]))

        with self.subTest(test="invalid list"):
            self.assertFalse(field.is_equal_to([6, 7]))

    def test_is_equal_to_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            _get_instance(True).get.id_.is_equal_to("")
        self.assertEqual("invalid 'other' type: str", exc.exception.args[0])


class TestOperationMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.operation,
            bytes_count=1,
            content=b"\x07",
            is_empty=False,
            name="f8",
            words_count=1,
            wo_attrs=["struct"]
        )

    def test_desc(self) -> None:
        self.assertEqual(
            Code.ERROR, _get_instance(True).get.operation.desc()
        )


class TestResponseMessageField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            _get_instance(True).get.response,
            bytes_count=1,
            content=b"\x08",
            is_empty=False,
            name="f9",
            words_count=1,
            wo_attrs=["struct"]
        )

    def test_desc(self) -> None:
        self.assertEqual(
            Code.DMA, _get_instance(True).get.response.desc()
        )


class TestMessageFieldPattern(unittest.TestCase):

    def test_basic(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                stop=None,
                bytes_expected=0,
                default=b"",
                typename="basic",
            ),
            MessageFieldPattern.basic(fmt=Code.U8).__init_kwargs__()
        )


    def test_single(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=1,
                default=b"",
                typename="single",
            ),
            MessageFieldPattern.single(fmt=Code.U8).__init_kwargs__()
        )

    def test_static(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=1,
                default=b"\x00",
                typename="static",
            ),
            MessageFieldPattern.static(
                fmt=Code.U8, default=b"\x00"
            ).__init_kwargs__()
        )

    def test_address(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=1,
                behaviour=Code.DMA,
                default=b"",
                typename="address",
            ),
            MessageFieldPattern.address(fmt=Code.U8).__init_kwargs__()
        )

    def test_crc(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U16,
                order=Code.BIG_ENDIAN,
                bytes_expected=2,
                default=b"",
                poly=0x1021,
                init=0,
                wo_fields=set(),
                typename="crc",
            ),
            MessageFieldPattern.crc(fmt=Code.U16).__init_kwargs__()
        )

    def test_data(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                stop=None,
                bytes_expected=0,
                default=b"",
                typename="data",
            ),
            MessageFieldPattern.data(fmt=Code.U8).__init_kwargs__()
        )

    def test_data_length(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=1,
                default=b"",
                behaviour=Code.ACTUAL,
                units=Code.BYTES,
                additive=0,
                typename="data_length",
            ),
            MessageFieldPattern.data_length(fmt=Code.U8).__init_kwargs__()
        )

    def test_id_(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=1,
                default=b"",
                typename="id",
            ),
            MessageFieldPattern.id_(fmt=Code.U8).__init_kwargs__()
        )

    def test_operations(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U8,
                order=Code.BIG_ENDIAN,
                bytes_expected=1,
                default=b"",
                descs={0: Code.READ, 1: Code.WRITE},
                typename="operation",
            ),
            MessageFieldPattern.operation(fmt=Code.U8).__init_kwargs__()
        )

    def test_response(self) -> None:
        self.assertDictEqual(
            dict(
                fmt=Code.U24,
                order=Code.BIG_ENDIAN,
                bytes_expected=3,
                default=b"",
                descs={},
                typename="response",
            ),
            MessageFieldPattern.response(fmt=Code.U24).__init_kwargs__()
        )
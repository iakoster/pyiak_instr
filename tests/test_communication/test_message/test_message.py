import unittest

import numpy as np

from ..utils import compare_objects

from pyinstr_iakoster.core import Code
from pyinstr_iakoster.communication import (
    Message,
    FieldSetter,
    SingleField,
    StaticField,
    AddressField,
    DataField,
    CrcField,
    DataLengthField,
    OperationField,
    ResponseField,
    MessageContentError
)


RESPONSE_CODES = {
    0: Code.OK,
    4: Code.WAIT
}


class AnotherMessage(Message):

    def __init__(
            self,
            mf_name: str = "new_def",
            splitable: bool = False,
            slice_length: int = 1024
    ):
        Message.__init__(
            self,
            mf_name=mf_name,
            splitable=splitable,
            slice_length=slice_length
        )


class TestMessage(unittest.TestCase):

    def fill_content(self) -> bytes:
        content = b"\x1a\xa5\x00\x00\xaa\x01\x04\xff\xee\xdd\xcc\x39\x86"
        self.msg.extract(content)
        return content

    def setUp(self) -> None:
        self.msg: Message = Message().configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=1, fmt=">I"),
            crc=FieldSetter.crc(fmt=">H"),
        )
        self.simple_msg: Message = Message().configure(
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(
                fmt=">B", units=FieldSetter.WORDS
            ),
            data=FieldSetter.data(expected=-1, fmt=">H")
        )

    def test_init(self):
        msg = Message()
        self.assertEqual("std", msg.mf_name)
        with self.assertRaises(KeyError) as exc:
            msg.data.unpack()
        self.assertEqual(
            "data", exc.exception.args[0]
        )

    def test_configure(self):
        msg = Message(mf_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=4, fmt=">I"),
            crc=FieldSetter.crc(fmt=">H"),
        )
        fields = dict(
            preamble=StaticField(
                "not_def",
                "preamble",
                start_byte=0,
                fmt=">H",
                default=0x1aa5,
                parent=msg,
            ),
            response=ResponseField(
                "not_def",
                "response",
                start_byte=2,
                fmt=">B",
                codes=RESPONSE_CODES,
                default_code=Code.UNDEFINED,
                parent=msg,
            ),
            address=AddressField(
                "not_def",
                start_byte=3,
                fmt=">H",
                parent=msg,
            ),
            operation=OperationField(
                "not_def",
                start_byte=5,
                fmt=">B",
                parent=msg,
            ),
            data_length=DataLengthField(
                "not_def",
                start_byte=6,
                fmt=">B",
                parent=msg,
            ),
            data=DataField(
                "not_def",
                start_byte=7,
                expected=4,
                fmt=">I",
                parent=msg,
            ),
            crc=CrcField(
                "not_def",
                "crc",
                start_byte=23,
                fmt=">H",
                parent=msg,
            )
        )
        for name, field in fields.items():
            compare_objects(self, field, msg[name])

    def test_configure_middle_infinite(self):
        msg = Message(mf_name="inf").configure(
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=-1, fmt=">H"),
            address=FieldSetter.address(fmt=">H"),
            footer=FieldSetter.single(fmt=">H")
        ).set(
            operation=1,
            data_length=6,
            data=[1, 23, 4],
            address=0xaa55,
            footer=0x42
        )
        fields = dict(
            operation=OperationField(
                "inf",
                start_byte=0,
                fmt=">B",
                parent=msg,
            ),
            data_length=DataLengthField(
                "inf",
                start_byte=1,
                fmt=">B",
                parent=msg,
            ),
            data=DataField(
                "inf",
                start_byte=2,
                expected=-1,
                fmt=">H",
                parent=msg,
            ),
            address=AddressField(
                "inf",
                start_byte=3,
                fmt=">H",
                parent=msg,
            ),
            footer=SingleField(
                "inf",
                "footer",
                start_byte=4,
                fmt=">H",
                parent=msg,
            )
        )
        for name, field in fields.items():
            fields[name].set(msg[name].content)

        fields["data"].stop_byte = -4
        fields["address"].start_byte = -4
        fields["address"].stop_byte = -2
        fields["footer"].start_byte = -2
        fields["footer"].stop_byte = None

        for name, field in fields.items():
            compare_objects(self, field, msg[name])
        self.assertEqual(
            b"\x01\x06\x00\x01\x00\x17\x00\x04\xaa\x55\x00\x42",
            msg.to_bytes()
        )
        self.assertEqual(
            b"".join(f.content for f in fields.values()),
            msg.to_bytes()
        )

    def test_extract(self):
        msg: Message = Message(mf_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.single(fmt=">B"),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=1, fmt=">I"),
            crc=FieldSetter.base(expected=1, fmt=">H"),
        )
        msg.extract(b"\x1a\xa5\x32\x01\x55\x01\x04\xff\xff\xf1\xfe\xee\xdd")
        for field, content in dict(
                preamble=b"\x1a\xa5",
                response=b"\x32",
                address=b"\x01\x55",
                operation=b"\x01",
                data_length=b"\x04",
                data=b"\xff\xff\xf1\xfe",
                crc=b"\xee\xdd",
        ).items():
            with self.subTest(field=field):
                self.assertEqual(content, msg[field].content)
        self.assertEqual("w", msg.operation.desc)

    def test_extract_middle_infinite(self):
        msg: Message = Message(mf_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=-1, fmt=">I"),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            crc=FieldSetter.base(expected=1, fmt=">H"),
        )
        msg.extract(
            b"\x1a\xa5\x32\x04\xff\xff\xf1\xfexAf3\x01\x55\x00\xee\xdd"
        )
        for field, content in dict(
                preamble=b"\x1a\xa5",
                response=b"\x32",
                address=b"\x01\x55",
                operation=b"\x00",
                data_length=b"\x04",
                data=b"\xff\xff\xf1\xfexAf3",
                crc=b"\xee\xdd",
        ).items():
            with self.subTest(field=field):
                self.assertEqual(content, msg[field].content)
        self.assertEqual("r", msg.operation.desc)
        self.assertEqual(
            b"\x1a\xa5\x32\x04\xff\xff\xf1\xfexAf3\x01\x55\x00\xee\xdd",
            msg.to_bytes()
        )

    def test_get_instance(self):
        msg: Message = self.msg.get_instance(mf_name="def")
        self.assertEqual("def", msg.mf_name)
        for _ in msg:
            self.assertFalse(True, "there is cannot be fields")

    def test_get_same_instance(self):
        msg: Message = self.msg.get_same_instance()
        self.assertEqual("std", msg.mf_name)
        for ref, res in zip(self.msg, msg):
            self.assertEqual(ref.name, res.name)
            self.assertIn(str(res), ("1AA5", "0", ""))

    def test_hex(self):
        self.fill_content()
        self.assertEqual("1aa5 00 00aa 01 04 ffeeddcc 3986", self.msg.hex())

    def test_to_bytes(self):
        content = self.fill_content()
        self.assertEqual(content, self.msg.to_bytes())

    def test_unpack(self):
        self.fill_content()

        self.assertTrue(
            (np.array([6821, 0, 170, 1, 4, 4293844428, 14726]) ==
             self.msg.unpack()).all()
        )

    def test_validate_content(self):
        with self.subTest(name="may_be_empty"):
            self.simple_msg.set(
                address=0x12,
                operation=0x01,
                data_length=2,
                data=[0x12, 0x3456]
            )
            self.assertListEqual(
                [0x12, 0x3456], list(self.simple_msg.data.unpack())
            )
            self.setUp()
            self.simple_msg.set(
                address=0x12,
                operation=0x00,
                data_length=2,
                data=[]
            )
            self.assertListEqual(
                [], list(self.simple_msg.data.unpack())
            )
            self.setUp()
            with self.assertRaises(MessageContentError) as exc:
                self.simple_msg.set(
                    address=0, operation=b"", data_length=1, data=[0x11]
                )
            self.assertEqual("Error with operation in Message: field is empty", exc.exception.args[0])

        with self.subTest(name="invalid data length"):
            with self.assertRaises(MessageContentError) as exc:
                self.simple_msg.set(
                    address=0, operation="w", data_length=1, data=[0, 1]
                )
            self.assertEqual(
                "Error with data_length in Message: invalid length",
                exc.exception.args[0]
            )

        with self.subTest(name="invalid crc"):
            with self.assertRaises(MessageContentError) as exc:
                self.msg.set(
                    preamble=0x1aa5,
                    response=0,
                    address=0x10,
                    operation="w",
                    crc=10,
                )
            self.assertIn(
                "Error with crc in Message: invalid crc value, ",
                exc.exception.args[0]
            )

    def test_empry_write(self):
        self.assertEqual(
            b"\x00\x11\x01\x0a",
            self.simple_msg.set(
                address=0x11,
                data_length=10,
                operation="w",
                data=[]
            ).to_bytes()
        )

    def test_split(self):

        def get_test_msg() -> Message:
            new_msg = Message(splitable=True, slice_length=64)
            new_msg.configure(
                preamble=FieldSetter.static(fmt=">H", default=0x102),
                address=FieldSetter.address(fmt=">I"),
                data_length=FieldSetter.data_length(fmt=">I"),
                test_field=FieldSetter.base(expected=1, fmt=">B"),
                operation=FieldSetter.operation(fmt=">I", desc_dict={"r": 0, "w": 1}),
                data=FieldSetter.data(expected=-1, fmt=">I")
            )
            return new_msg

        def get_mess(addr: int, oper: int, data_len: int,
                     data=None, data_dim=FieldSetter.WORDS):
            if data is None:
                data = b''
            new_msg = get_test_msg()
            new_msg._max_data_len = 64
            new_msg.data_length._units = data_dim
            new_msg.set(
                preamble=0x102, address=addr,
                data_length=data_len, test_field=0xff,
                operation=oper, data=data)
            return new_msg

        test = (
            get_mess(0, 0, 34),
            get_mess(0, 0, 64),
            get_mess(0, 0, 128),
            get_mess(0, 0, 127),
            get_mess(0, 1, 34, range(34)),
            get_mess(0, 1, 64, range(64)),
            get_mess(0, 1, 128, range(128)),
            get_mess(0, 1, 127, range(127)),
            get_mess(0, 1, 124, range(31), data_dim=FieldSetter.BYTES),
        )
        expected = (
            (get_mess(0, 0, 34),),
            (get_mess(0, 0, 64),),
            (get_mess(0, 0, 64), get_mess(64, 0, 64)),
            (get_mess(0, 0, 64), get_mess(64, 0, 63)),
            (get_mess(0, 1, 34, range(34)),),
            (get_mess(0, 1, 64, range(64)),),
            (
                get_mess(0, 1, 64, range(64)),
                get_mess(64, 1, 64, range(64, 128))
            ), (
                get_mess(0, 1, 64, range(64)),
                get_mess(64, 1, 63, range(64, 127))
            ), (
                get_mess(0, 1, 64, range(16), data_dim=FieldSetter.BYTES),
                get_mess(64, 1, 60, range(16, 31), data_dim=FieldSetter.BYTES)
            ),
        )
        for i_mess, test_msg in enumerate(test):
            for i_part, mess in enumerate(test_msg.split()):
                with self.subTest(i_mess=i_mess, i_part=i_part):
                    self.assertEqual(str(expected[i_mess][i_part]), str(mess))
                    self.assertEqual(
                        expected[i_mess][i_part].mf_name, mess.mf_name
                    )

    def test_magic_bytes(self):
        content = self.fill_content()
        self.assertEqual(content, bytes(self.msg))

    def test_magic_str(self):
        self.fill_content()
        self.assertEqual("1AA5 0 AA 1 4 FFEEDDCC 3986", str(self.msg))

    def test_magic_len(self):
        content = self.fill_content()
        self.assertEqual(len(content), len(self.msg))

    def test_magic_repr(self):
        self.fill_content()
        self.assertEqual(
            "<Message(preamble=1AA5, response=0, address=AA, operation=1, "
            "data_length=4, data=FFEEDDCC, crc=3986), src=None, dst=None>",
            repr(self.msg)
        )
        self.msg.set_src_dst(src="COM4", dst=("192.168.0.1", 3202))
        self.assertEqual(
            "<Message(preamble=1AA5, response=0, address=AA, operation=1, "
            "data_length=4, data=FFEEDDCC, crc=3986), "
            "src=COM4, dst=('192.168.0.1', 3202)>",
            repr(self.msg)
        )

    def test_magic_add_bytes(self):
        self.fill_content()
        self.msg += b"\x01\x02\x03\x04"
        self.assertEqual(
            b"\xff\xee\xdd\xcc\x01\x02\x03\x04", self.msg.data.content
        )
        self.assertEqual(2, self.msg.data.expected)
        self.assertEqual(8, self.msg.data_length.unpack())

    def test_magic_add_message(self):
        self.fill_content()
        msg = self.msg.get_same_instance()
        msg.set(
            preamble=0x1aa5,
            response=0,
            address=0x1234,
            operation=1,
            data_length=4,
            data=0x1234,
            crc=255,
        )
        self.msg += msg
        self.assertEqual(
            b"\xff\xee\xdd\xcc\x00\x00\x12\x34", self.msg.data.content
        )
        self.assertEqual(2, self.msg.data.expected)
        self.assertEqual(8, self.msg.data_length.unpack())

    def test_madic_add_errors(self):
        with self.assertRaises(TypeError) as exc:
            self.msg += self.msg.get_instance(mf_name="new")
        self.assertEqual(
            "messages have different formats: new != std",
            exc.exception.args[0]
        )

        with self.assertRaises(TypeError) as exc:
            self.msg += 1
        self.assertIn(
            "cannot be added to the message", exc.exception.args[0]
        )

    def test_magic_add_class(self):
        msg_1 = AnotherMessage().configure(
            address=FieldSetter.address(fmt=">B"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=-1, fmt=">B"),
        )
        msg_2 = msg_1.get_same_instance()
        for msg in (msg_1, msg_2):
            msg.set(
                address=0x1a,
                operation=1,
                data_length=1,
                data=0xee,
            )

        msg_1 += msg_2
        self.assertEqual(b"\xee\xee", msg_1.data.content)
        self.assertEqual(-1, msg_1.data.expected)
        self.assertEqual(2, msg_1.data_length.unpack())


class TestFields(unittest.TestCase):

    def test_data_length_unpdate(self):
        msg: Message = Message(mf_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=1, fmt=">I"),
            crc=FieldSetter.base(expected=1, fmt=">H"),
        )
        msg.extract(b"\x1a\xa5\x32\x01\x55\x01\x04\xff\xff\xf1\xfe\xee\xdd")
        dlen = DataLengthField("def", start_byte=0, fmt=">H")
        self.assertListEqual([], list(dlen.unpack()))
        dlen.set(dlen.calculate(msg.data))
        self.assertListEqual([4], list(dlen.unpack()))

    def test_operation_compare(self):
        msg: Message = Message(mf_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=1, fmt=">I"),
            crc=FieldSetter.base(expected=1, fmt=">H"),
        )
        msg.extract(b"\x1a\xa5\x32\x01\x55\x01\x04\xff\xff\xf1\xfe\xee\xdd")
        oper = OperationField("def", start_byte=0, fmt=">H")
        self.assertFalse(oper.compare(msg))
        oper.set("w")
        self.assertTrue(oper.compare(msg))
        oper.set("r")
        self.assertFalse(oper.compare(msg))

import unittest
from typing import Any

import numpy as np

from pyinstr_iakoster.communication import (
    Message,
    FieldSetter,
    Field,
    SingleField,
    StaticField,
    AddressField,
    DataField,
    DataLengthField,
    OperationField,
    MessageContentError
)


class AnotherMessage(Message):

    def __init__(
            self,
            format_name: str = "new_def",
            splitable: bool = False,
            slice_length: int = 1024
    ):
        Message.__init__(
            self,
            format_name=format_name,
            splitable=splitable,
            slice_length=slice_length
        )


class TestFieldSetter(unittest.TestCase):

    def validate_setter(
            self,
            fs: FieldSetter,
            kwargs: dict,
            special: str = None
    ):
        self.assertDictEqual(kwargs, fs.kwargs)
        self.assertEqual(special, fs.special)

    def test_init(self):
        self.validate_setter(
            FieldSetter(a=0, b=3),
            {"a": 0, "b": 3}
        )

    def test_base(self):
        self.validate_setter(
            FieldSetter.base(expected=1, fmt="i"),
            {
                "expected": 1,
                "fmt": "i",
                "default": [],
                "info": None,
                'may_be_empty': False
            }
        )

    def test_single(self):
        self.validate_setter(
            FieldSetter.single(fmt="i"),
            {
                "fmt": "i",
                "default": [],
                "info": None,
                'may_be_empty': False
            },
            special="single"
        )

    def test_static(self):
        self.validate_setter(
            FieldSetter.static(fmt="i", default=[]),
            {"fmt": "i", "default": [], "info": None},
            special="static"
        )

    def test_address(self):
        self.validate_setter(
            FieldSetter.address(fmt="i"),
            {"fmt": "i", "info": None}
        )

    def test_data(self):
        self.validate_setter(
            FieldSetter.data(expected=3, fmt="i"),
            {"expected": 3, "fmt": "i", "info": None}
        )

    def test_data_length(self):
        self.validate_setter(
            FieldSetter.data_length(fmt="i"),
            {"fmt": "i", "additive": 0, "info": None, "units": 16}
        )

    def test_operation(self):
        self.validate_setter(
            FieldSetter.operation(fmt="i"),
            {"fmt": "i", "desc_dict": None, "info": None}
        )


class TestMessage(unittest.TestCase):

    def validate_field(
            self,
            field: Field,
            slice_: slice,
            field_class: type,
            **attributes: Any
    ):
        for name, val in attributes.items():
            with self.subTest(class_=field.__class__.__name__, name=name):
                self.assertEqual(val, field.__getattribute__(name))
        with self.subTest(class_=field.__class__.__name__, name="slice"):
            self.assertEqual(slice_.start, field.slice.start)
            self.assertEqual(slice_.stop, field.slice.stop)
        with self.subTest(
                class_=field.__class__.__name__, name="field_class"
        ):
            self.assertIs(field_class, field.field_class)
        with self.subTest(name="parent"):
            self.assertIsInstance(field.parent, Message)

    def fill_content(self) -> bytes:
        content = b"\x1a\xa5\x00\x00\xaa\x01\x04\xff\xee\xdd\xcc\x39\x86"
        self.msg.extract(content)
        return content

    def setUp(self) -> None:
        self.msg: Message = Message().configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.single(fmt=">B"),
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
        self.assertEqual("default", msg.format_name)
        with self.assertRaises(KeyError) as exc:
            msg.data.unpack()
        self.assertEqual(
            "data", exc.exception.args[0]
        )

    def test_configure(self):
        msg = Message(format_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.single(fmt=">B"),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=4, fmt=">I"),
            crc=FieldSetter.base(expected=2, fmt=">H"),
        )
        self.validate_field(
            msg["preamble"],
            slice(0, 2),
            StaticField,
            bytesize=2,
            content=b"\x1a\xa5",
            stop_byte=2,
            expected=1,
            finite=True,
            fmt=">H",
            info={},
            name="preamble",
            format_name="not_def",
            start_byte=0,
            words_count=1
        )
        self.validate_field(
            msg["response"],
            slice(2, 3),
            SingleField,
            bytesize=1,
            content=b"",
            stop_byte=3,
            expected=1,
            finite=True,
            fmt=">B",
            info={},
            name="response",
            format_name="not_def",
            start_byte=2,
            words_count=0
        )
        self.validate_field(
            msg["address"],
            slice(3, 5),
            AddressField,
            bytesize=2,
            content=b"",
            stop_byte=5,
            expected=1,
            finite=True,
            fmt=">H",
            info={},
            name="address",
            format_name="not_def",
            start_byte=3,
            words_count=0
        )
        self.validate_field(
            msg["operation"],
            slice(5, 6),
            OperationField,
            bytesize=1,
            content=b"",
            stop_byte=6,
            expected=1,
            finite=True,
            fmt=">B",
            info={},
            name="operation",
            format_name="not_def",
            start_byte=5,
            words_count=0,
            base="",
            desc="",
            desc_dict={"r": 0, "w": 1, "e": 2},
            desc_dict_r={0: "r", 1: "w", 2: "e"},
        )
        self.validate_field(
            msg["data_length"],
            slice(6, 7),
            DataLengthField,
            bytesize=1,
            content=b"",
            stop_byte=7,
            expected=1,
            finite=True,
            fmt=">B",
            info={},
            name="data_length",
            format_name="not_def",
            start_byte=6,
            words_count=0,
            units=0x10,
            additive=0
        )
        self.validate_field(
            msg["data"],
            slice(7, 23),
            DataField,
            bytesize=4,
            content=b"",
            stop_byte=23,
            expected=4,
            finite=True,
            fmt=">I",
            info={},
            name="data",
            format_name="not_def",
            start_byte=7,
            words_count=0,
        )
        self.validate_field(
            msg["crc"],
            slice(23, 27),
            Field,
            bytesize=2,
            content=b"",
            stop_byte=27,
            expected=2,
            finite=True,
            fmt=">H",
            info={},
            name="crc",
            format_name="not_def",
            start_byte=23,
            words_count=0,
        )

    def test_configure_middle_infinite(self):
        msg = Message(format_name="inf").configure(
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
        self.validate_field(
            msg["operation"],
            slice(0, 1),
            OperationField,
            bytesize=1,
            content=b"\x01",
            stop_byte=1,
            expected=1,
            finite=True,
            fmt=">B",
            info={},
            name="operation",
            format_name="inf",
            start_byte=0,
            words_count=1
        )
        self.validate_field(
            msg["data_length"],
            slice(1, 2),
            DataLengthField,
            bytesize=1,
            content=b"\x06",
            stop_byte=2,
            expected=1,
            finite=True,
            fmt=">B",
            info={},
            name="data_length",
            format_name="inf",
            start_byte=1,
            words_count=1
        )
        self.validate_field(
            msg["data"],
            slice(2, -4),
            DataField,
            bytesize=2,
            content=b"\x00\x01\x00\x17\x00\x04",
            stop_byte=-4,
            expected=-1,
            finite=False,
            fmt=">H",
            info={},
            name="data",
            format_name="inf",
            start_byte=2,
            words_count=3
        )
        self.validate_field(
            msg["address"],
            slice(-4, -2),
            AddressField,
            bytesize=2,
            content=b"\xaa\x55",
            stop_byte=-2,
            expected=1,
            finite=True,
            fmt=">H",
            info={},
            name="address",
            format_name="inf",
            start_byte=-4,
            words_count=1
        )
        self.validate_field(
            msg["footer"],
            slice(-2, None),
            SingleField,
            bytesize=2,
            content=b"\x00\x42",
            stop_byte=None,
            expected=1,
            finite=True,
            fmt=">H",
            info={},
            name="footer",
            format_name="inf",
            start_byte=-2,
            words_count=1
        )
        self.assertEqual(
            b"\x01\x06\x00\x01\x00\x17\x00\x04\xaa\x55\x00\x42",
            msg.to_bytes()
        )

    def test_extract(self):
        msg: Message = Message(format_name="not_def").configure(
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
        msg: Message = Message(format_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.single(fmt=">B"),
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
        msg: Message = self.msg.get_instance(format_name="def")
        self.assertEqual("def", msg.format_name)
        for _ in msg:
            self.assertFalse(True, "there is cannot be fields")

    def test_get_same_instance(self):
        msg: Message = self.msg.get_same_instance()
        self.assertEqual("default", msg.format_name)
        for ref, res in zip(self.msg, msg):
            self.assertEqual(ref.name, res.name)
            self.assertIn(str(res), ("1AA5", ""))

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
                        expected[i_mess][i_part].format_name, mess.format_name
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
            "data_length=4, data=FFEEDDCC, crc=3986), from=None, to=None>",
            repr(self.msg)
        )
        self.msg.set_addresses(tx=("192.168.0.1", 3202), rx="COM4")
        self.assertEqual(
            "<Message(preamble=1AA5, response=0, address=AA, operation=1, "
            "data_length=4, data=FFEEDDCC, crc=3986), "
            "from=192.168.0.1:3202, to=COM4>",
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
            self.msg += self.msg.get_instance(format_name="new")
        self.assertEqual(
            "messages have different formats: new != default",
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
        msg: Message = Message(format_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.single(fmt=">B"),
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
        msg: Message = Message(format_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.single(fmt=">B"),
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

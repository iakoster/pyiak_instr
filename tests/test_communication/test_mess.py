import unittest
from typing import Any

import numpy as np

from pyinstr_iakoster.communication import (
    Message,
    FieldSetter,
    Field,
    FieldSingle,
    FieldStatic,
    FieldAddress,
    FieldData,
    FieldDataLength,
    FieldOperation,
)


class TestFieldSetter(unittest.TestCase):

    def validate_setter(
            self,
            fs: FieldSetter,
            args: tuple,
            kwargs: dict,
            special: str = None
    ):
        self.assertTupleEqual(args, fs.args)
        self.assertDictEqual(kwargs, fs.kwargs)
        self.assertEqual(special, fs.special)

    def test_init(self):
        self.validate_setter(
            FieldSetter(1, 2, a=0, b=3),
            (1, 2),
            {"a": 0, "b": 3}
        )

    def test_base(self):
        self.validate_setter(
            FieldSetter.base(1, "i"),
            (1, "i"),
            {"content": b"", "info": None}
        )

    def test_single(self):
        self.validate_setter(
            FieldSetter.single("i"),
            ("i",),
            {"content": b"", "info": None},
            special="single"
        )

    def test_static(self):
        self.validate_setter(
            FieldSetter.static("i", b""),
            ("i", b""),
            {"info": None},
            special="static"
        )

    def test_address(self):
        self.validate_setter(
            FieldSetter.address("i"),
            ("i",),
            {"content": b"", "info": None}
        )

    def test_data(self):
        self.validate_setter(
            FieldSetter.data(3, "i"),
            (3, "i"),
            {"content": b"", "info": None}
        )

    def test_data_length(self):
        self.validate_setter(
            FieldSetter.data_length("i"),
            ("i",),
            {"additive": 0, "content": b"", "info": None, "units": 16}
        )

    def test_operation(self):
        self.validate_setter(
            FieldSetter.operation("i"),
            ("i",),
            {"content": b"", "desc_dict": None, "info": None}
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
            with self.subTest(name=name):
                self.assertEqual(val, field.__getattribute__(name))
        with self.subTest(name="slice"):
            self.assertEqual(slice_.start, field.slice.start)
            self.assertEqual(slice_.stop, field.slice.stop)
        with self.subTest(name="field_class"):
            self.assertIs(field_class, field.field_class)

    def fill_content(self) -> bytes:
        content = b"\x1a\xa5\x00\x00\xaa\x01\x04\xff\xee\xdd\xcc\x12\x54"
        self.msg.extract(content)
        return content

    def setUp(self) -> None:
        self.msg = Message().configure(
            preamble=FieldSetter.static(">H", 0x1aa5),
            response=FieldSetter.single(">B"),
            address=FieldSetter.address(">H"),
            operation=FieldSetter.operation(">B"),
            data_length=FieldSetter.data_length(">B"),
            data=FieldSetter.data(1, ">I"),
            crc=FieldSetter.base(1, ">H"),
        )

    def test_init(self):
        msg = Message()
        self.assertEqual("default", msg.format_name)
        with self.assertRaises(AttributeError) as exc:
            msg.data.unpack()
        self.assertIn(
            "'Message' object has no attribute", exc.exception.args[0]
        )

    def test_configure(self):
        msg = Message(format_name="not_def").configure(
            preamble=FieldSetter.static(">H", 0x1aa5),
            response=FieldSetter.single(">B"),
            address=FieldSetter.address(">H"),
            operation=FieldSetter.operation(">B"),
            data_length=FieldSetter.data_length(">B"),
            data=FieldSetter.data(4, ">I"),
            crc=FieldSetter.base(2, ">H"),
        )
        self.validate_field(
            msg["preamble"],
            slice(0, 2),
            FieldStatic,
            bytesize=2,
            content=b"\x1a\xa5",
            end_byte=2,
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
            FieldSingle,
            bytesize=1,
            content=b"",
            end_byte=3,
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
            FieldAddress,
            bytesize=2,
            content=b"",
            end_byte=5,
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
            FieldOperation,
            bytesize=1,
            content=b"",
            end_byte=6,
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
            desc_dict_rev={0: "r", 1: "w", 2: "e"},
        )
        self.validate_field(
            msg["data_length"],
            slice(6, 7),
            FieldDataLength,
            bytesize=1,
            content=b"",
            end_byte=7,
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
            FieldData,
            bytesize=4,
            content=b"",
            end_byte=23,
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
            end_byte=27,
            expected=2,
            finite=True,
            fmt=">H",
            info={},
            name="crc",
            format_name="not_def",
            start_byte=23,
            words_count=0,
        )

    def test_extract(self):
        msg: Message = Message(format_name="not_def").configure(
            preamble=FieldSetter.static(">H", 0x1aa5),
            response=FieldSetter.single(">B"),
            address=FieldSetter.address(">H"),
            operation=FieldSetter.operation(">B"),
            data_length=FieldSetter.data_length(">B"),
            data=FieldSetter.data(1, ">I"),
            crc=FieldSetter.base(1, ">H"),
        )
        msg.extract(b"\x1a\xa5\x32\x01\x55\x01\x01\xff\xff\xf1\xfe\xee\xdd")
        for field, content in dict(
                preamble=b"\x1a\xa5",
                response=b"\x32",
                address=b"\x01\x55",
                operation=b"\x01",
                data_length=b"\x01",
                data=b"\xff\xff\xf1\xfe",
                crc=b"\xee\xdd",
        ).items():
            with self.subTest(field=field):
                self.assertEqual(content, msg[field].content)
        self.assertEqual("w", msg.operation.desc)

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
        self.assertEqual("1aa5 00 00aa 01 04 ffeeddcc 1254", self.msg.hex())

    def test_to_bytes(self):
        content = self.fill_content()
        self.assertEqual(content, self.msg.to_bytes())

    def test_unpack(self):
        self.fill_content()

        self.assertTrue(
            (np.array([6821, 0, 170, 1, 4, 4293844428, 4692]) ==
             self.msg.unpack()).all()
        )

    def test_magic_bytes(self):
        content = self.fill_content()
        self.assertEqual(content, bytes(self.msg))

    def test_magic_str(self):
        self.fill_content()
        self.assertEqual("1AA5 0 AA 1 4 FFEEDDCC 1254", str(self.msg))

    def test_magic_len(self):
        content = self.fill_content()
        self.assertEqual(len(content), len(self.msg))

    def test_magic_repr(self):
        self.fill_content()
        self.assertEqual(
            "<Message(preamble=1AA5, response=0, address=AA, operation=1, "
            "data_length=4, data=FFEEDDCC, crc=1254), from=None, to=None>",
            repr(self.msg)
        )
        self.msg.set_addresses(tx=("192.168.0.1", 3202), rx="COM4")
        self.assertEqual(
            "<Message(preamble=1AA5, response=0, address=AA, operation=1, "
            "data_length=4, data=FFEEDDCC, crc=1254), "
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
        msg.set_fields_content(
            preamble=0x1aa5,
            response=0,
            address=0x1234,
            operation=1,
            data_length=2,
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

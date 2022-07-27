import unittest

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.communication import (
    FieldSetter,
    Message,
    MessageErrorMark,
    MessageFormat,
    PackageFormat
)


DATA_TEST_PATH = DATA_TEST_DIR / "test.json"


def get_mf_asm(reference: bool = True):

    mf = MessageFormat(
        emark=MessageErrorMark(
            operation="neq",
            start_byte=12,
            stop_byte=16,
            value=b"\x00\x00\x00\x01"
        ),
        format_name="asm",
        splitable=True,
        slice_length=1024,
        address=FieldSetter.address(fmt=">I"),
        data_length=FieldSetter.data_length(
            fmt=">I", units=FieldSetter.WORDS
        ),
        operation=FieldSetter.operation(
            fmt=">I", desc_dict={"w": 0, "r": 1}
        ),
        data=FieldSetter.data(expected=-1, fmt=">I")
    )

    if reference:
        return mf, dict(
            msg_args=dict(
                format_name="asm", splitable=True, slice_length=1024
            ),
            setters=dict(
                address=dict(special=None, kwargs=dict(fmt=">I", info=None)),
                data_length=dict(special=None, kwargs=dict(
                    fmt=">I", units=0x11, info=None, additive=0,
                )),
                operation=dict(special=None, kwargs=dict(
                    fmt=">I", desc_dict={"w": 0, "r": 1}, info=None
                )),
                data=dict(special=None, kwargs=dict(
                    expected=-1, fmt=">I", info=None
                ))
            )
        )
    return mf


def get_mf_kpm(reference: bool = True):

    mf = MessageFormat(
        emark=MessageErrorMark(
            operation="neq", field_name="response", value=[0]
        ),
        format_name="kpm",
        splitable=False,
        slice_length=1024,
        preamble=FieldSetter.static(fmt=">H", default=0xaa55),
        operation=FieldSetter.operation(
            fmt=">B", desc_dict={
                "wp": 1, "rp": 2, "wn": 3, "rn": 4
            }
        ),
        response=FieldSetter.single(fmt=">B"),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=">f"),
        crc=FieldSetter.crc(fmt=">H")
    )

    if reference:
        return mf, dict(
            msg_args=dict(
                format_name="kpm", splitable=False, slice_length=1024
            ),
            setters=dict(
                preamble=dict(special="static", kwargs=dict(
                    fmt=">H", default=0xaa55, info=None
                )),
                operation=dict(special=None, kwargs=dict(
                    fmt=">B",
                    desc_dict={"wp": 1, "rp": 2, "wn": 3, "rn": 4},
                    info=None
                )),
                response=dict(special="single", kwargs=dict(
                    fmt=">B", default=[], info=None, may_be_empty=False,
                )),
                address=dict(special=None, kwargs=dict(fmt=">H", info=None)),
                data_length=dict(special=None, kwargs=dict(
                    fmt=">H", units=0x10, info=None, additive=0,
                )),
                data=dict(special=None, kwargs=dict(
                    expected=-1, fmt=">f", info=None
                )),
                crc=dict(special="crc", kwargs=dict(
                    fmt=">H", algorithm_name="crc16-CCITT XMODEM", info=None
                ))
            )
        )
    return mf


class TestMessageErrorMark(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.emarks = {
            "empty": MessageErrorMark(),
            0: MessageErrorMark(
                operation="eq",
                value=b"\x4c",
                start_byte=0,
                stop_byte=1,
            ),
            1: MessageErrorMark(
                operation="neq",
                value=b"\x4c\x54",
                start_byte=1,
                stop_byte=3,
            ),
            2: MessageErrorMark(
                operation="eq",
                value=b"\x4c",
                field_name="response"
            ),
            3: MessageErrorMark(
                operation="neq",
                value=[7],
                field_name="response"
            ),
        }

    @staticmethod
    def get_message(content: bytes) -> Message:
        return Message().configure(
            address=FieldSetter.address(fmt="B"),
            response=FieldSetter.single(fmt="B", default=0),
            operation=FieldSetter.operation(fmt="B"),
            data_length=FieldSetter.data_length(fmt="B"),
            data=FieldSetter.data(expected=-1, fmt="B")
        ).extract(content)

    def test_emarks(self):
        msgs = (
            b"\x4c\x4c\x54\x01\xfe",
            b"\xef\x07\xaa\x01\xed",
            b"\x12\x4c\x54\x01\x9a"
        )
        ret_msg = {
            "empty": msgs,
            0: tuple(m[1:] for m in msgs),
            1: tuple(m[:1] + m[3:] for m in msgs),
            2: msgs,
            3: msgs,
        }
        ret_emark = {
            "empty": (False, False, False),
            0: (True, False, False),
            1: (False, True, False),
            2: (True, False, True),
            3: (True, False, True),
        }
        for key, emark in self.emarks.items():
            for i_msg, msg in enumerate(msgs):
                if not emark.bytes_required:
                    msg = self.get_message(msg)
                res_msg, res = emark.match(msg)
                if not emark.bytes_required:
                    res_msg = res_msg.to_bytes()

                with self.subTest(key=key, i_msg=i_msg, name="emark"):
                    self.assertEqual(ret_emark[key][i_msg], res)
                with self.subTest(key=key, i_msg=i_msg, name="msg"):
                    self.assertEqual(ret_msg[key][i_msg], res_msg)

    def test_operation_not_exists(self):
        with self.assertRaises(ValueError) as exc:
            MessageErrorMark(operation="lol")
        self.assertEqual(
            "'lol' operation not in {'eq', 'neq'}", exc.exception.args[0]
        )

    def test_invalid_range(self):
        with self.assertRaises(ValueError) as exc:
            MessageErrorMark(operation="eq", start_byte=10)
        self.assertEqual(
            "field name or start byte and end byte must be defined",
            exc.exception.args[0]
        )

    def test_value_not_exists(self):
        with self.assertRaises(ValueError) as exc:
            MessageErrorMark(operation="eq", field_name="req")
        self.assertEqual("value not specified", exc.exception.args[0])

    def test_value_not_bytes(self):
        with self.assertRaises(TypeError) as exc:
            MessageErrorMark(
                operation="eq", start_byte=1, stop_byte=2, value=[10]
            )
        self.assertEqual(
            "if start and end bytes is specified that value must be bytes",
            exc.exception.args[0]
        )

    def test_match_not_bytes(self):
        with self.assertRaises(TypeError) as exc:
            self.emarks[0].match(self.get_message(b"12345"))
        self.assertEqual("bytes type required", exc.exception.args[0])


class TestMessageFormat(unittest.TestCase):

    def test_init(self):

        for mf, ref_data in (get_mf_asm(), get_mf_kpm()):
            format_name = mf.msg_args["format_name"]

            with self.subTest(format_name=format_name):
                self.assertDictEqual(ref_data["msg_args"], mf.msg_args)

            with self.subTest(format_name=format_name, setter="all"):
                self.assertEqual(len(ref_data["setters"]), len(mf.setters))
                for (ref_name, ref_setter), (name, setter) in zip(
                    ref_data["setters"].items(), mf.setters.items()
                ):
                    with self.subTest(format_name=format_name, setter=name):
                        self.assertEqual(ref_name, name)
                        self.assertEqual(
                            ref_setter["special"], setter.special
                        )
                        self.assertDictEqual(
                            ref_setter["kwargs"], setter.kwargs
                        )


class TestPackageFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.pf = PackageFormat(
            asm=get_mf_asm(False),
            kpm=get_mf_kpm(False)
        )

    def test_write_read(self):
        self.pf.write(DATA_TEST_PATH)
        pf = PackageFormat.read(DATA_TEST_PATH)
        for name, ref_mf in self.pf.formats.items():
            mf = pf[name]
            with self.subTest(name=name):
                self.assertEqual(ref_mf.msg_args, mf.msg_args)

            with self.subTest(name):
                self.assertDictEqual(ref_mf.emark.kwargs, mf.emark.kwargs)

            with self.subTest(name=name, setter="all"):
                self.assertEqual(len(ref_mf.setters), len(mf.setters))
                for (ref_set_name, ref_setter), (set_name, setter) in zip(
                    ref_mf.setters.items(), mf.setters.items()
                ):
                    with self.subTest(name=name, setter=name):
                        self.assertEqual(name, name)
                        self.assertEqual(ref_setter.special, setter.special)

                        self.assertDictEqual(
                            {k: v for k, v in ref_setter.kwargs.items()
                             if v is not None},
                            setter.kwargs
                        )

    def test_get_asm_basic(self):
        asm_msg: Message = Message(
            format_name="asm", splitable=True
        ).configure(
            address=FieldSetter.address(fmt=">I"),
            data_length=FieldSetter.data_length(
                fmt=">I", units=FieldSetter.WORDS
            ),
            operation=FieldSetter.operation(
                fmt=">I", desc_dict={"w": 0, "r": 1}
            ),
            data=FieldSetter.data(expected=-1, fmt=">I")
        ).set(
            address=0x01020304,
            data_length=2,
            operation="w",
            data=[34, 52]
        )
        message = self.pf.get("asm").extract(
            b"\x01\x02\x03\x04\x00\x00\x00\x02\x00\x00\x00\x00"
            b"\x00\x00\x00\x22\x00\x00\x00\x34"
        )
        self.assertEqual(asm_msg.to_bytes(), message.to_bytes())
        for ref_field, field in zip(asm_msg, message):
            with self.subTest(ref=ref_field.name):
                self.assertEqual(ref_field.name, field.name)
                self.assertEqual(ref_field.content, field.content)

    def test_get_kpm_basic(self):
        kpm_msg: Message = Message(format_name="kpm").configure(
            preamble=FieldSetter.static(fmt=">H", default=0xaa55),
            operation=FieldSetter.operation(
                fmt=">B", desc_dict={
                    "wp": 1, "rp": 2, "wn": 3, "rn": 4
                }
            ),
            response=FieldSetter.single(fmt=">B"),
            address=FieldSetter.address(fmt=">H"),
            data_length=FieldSetter.data_length(fmt=">H"),
            data=FieldSetter.data(expected=-1, fmt=">b"),
            crc=FieldSetter.single(fmt=">H")
        ).set(
            operation="wp",
            response=0,
            address=0x33,
            data_length=2,
            data=[17, 32],
            crc=0xedbc
        )
        message = self.pf.get("kpm", data={"fmt": ">b"})
        message.extract(
            b"\xaa\x55\x01\x00\x00\x33\x00\x02\x11\x20\xed\xbc"
        )
        self.assertEqual(bytes(kpm_msg), bytes(message))
        for ref_field, field in zip(kpm_msg, message):
            self.assertEqual(ref_field.name, field.name)
            self.assertEqual(ref_field.content, field.content)

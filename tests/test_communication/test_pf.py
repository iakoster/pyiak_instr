import unittest

import pandas.testing

from tests.env_vars import DATA_TEST_DIR
from .utils import (
    get_asm_msg,
    get_kpm_msg,
    get_mf_asm,
    get_mf_kpm,
    get_register_map_data,
    compare_registers,
    compare_messages,
)

from pyinstr_iakoster.communication import (
    FieldSetter,
    RegisterMap,
    Register,
    Message,
    MessageErrorMark,
    MessageFormat,
    PackageFormat
)


DATA_JSON_PATH = DATA_TEST_DIR / "test.json"
DATA_DB_PATH = DATA_TEST_DIR / "test.db"


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
                res_msg, res = emark.exists(msg)
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
            "field name or start and stop bytes must be defined",
            exc.exception.args[0]
        )

    def test_value_not_bytes(self):
        with self.assertRaises(TypeError) as exc:
            MessageErrorMark(
                operation="eq", start_byte=1, stop_byte=2, value=[10]
            )
        self.assertEqual(
            "invalid type: <class 'list'>, expected bytes",
            exc.exception.args[0]
        )

    def test_match_not_bytes(self):
        with self.assertRaises(TypeError) as exc:
            self.emarks[0].exists(self.get_message(b"12345"))
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

    REG_MAP_DATA = get_register_map_data()

    def setUp(self) -> None:
        self.pf = PackageFormat(
            register_map=RegisterMap(self.REG_MAP_DATA),
            asm=get_mf_asm(False),
            kpm=get_mf_kpm(False)
        )

    def test_write_read(self):
        self.pf.write(DATA_JSON_PATH, DATA_DB_PATH)
        pf = PackageFormat.read(DATA_JSON_PATH)\
            .read_register_map(DATA_DB_PATH)
        for name, ref_mf in self.pf.formats.items():
            mf = pf.get_format(name)
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

        with self.subTest(test="register_map"):
            pandas.testing.assert_frame_equal(
                pf.register_map.table,
                self.pf.register_map.table,
                check_names=False
            )

    def test_get_asm_basic(self):
        ref = get_asm_msg().set(
            address=0x01020304,
            data_length=2,
            operation="w",
            data=[34, 52]
        )
        res = self.pf.get("asm").extract(
            b"\x01\x02\x03\x04\x00\x00\x00\x02\x00\x00\x00\x00"
            b"\x00\x00\x00\x22\x00\x00\x00\x34"
        )
        compare_messages(self, ref, res)

    def test_get_kpm_basic(self):
        ref = get_kpm_msg().set(
            operation="wp",
            response=0,
            address=0x33,
            data_length=2,
            data=[17, 32],
            crc=0xedbc
        )
        res = self.pf.get("kpm", data={"fmt": ">b"}).extract(
            b"\xaa\x55\x01\x00\x00\x33\x00\x02\x11\x20\xed\xbc"
        )
        compare_messages(self, ref, res)

    def test_get_register(self):
        res = self.pf.get_register("tst_4")
        compare_registers(
            self,
            Register(
                "tst_4",
                "test_4",
                "asm",
                0x1000,
                7,
                "rw",
                description="test address 4. Other description."
            ),
            res
        )
        compare_messages(
            self,
            get_asm_msg().set(
                address=0x1000,
                data_length=1,
                operation=0,
                data=10
            ),
            res.write([10])
        )

    def test_getattr(self):
        compare_messages(
            self,
            get_kpm_msg(data_fmt=">H").set(
                address=0xf000,
                operation="rp",
                data_length=6,
                data=[3, 11, 32]
            ),
            self.pf.test_6.read(
                data=[3, 11, 32], update={"data": {"fmt": ">H"}}
            )
        )

    def test_write_with_update(self):
        self.assertEqual(
            ">f",
            self.pf.test_0.read(update={"data": {"fmt": ">f"}}).data.fmt
        )

    def test_read_wo(self):
        with self.assertRaises(TypeError) as exc:
            self.pf.test_2.read()
        self.assertEqual(
            "writing only", exc.exception.args[0]
        )

    def test_write_ro(self):
        with self.assertRaises(TypeError) as exc:
            self.pf.test_1.write()
        self.assertEqual(
            "reading only", exc.exception.args[0]
        )

    def test_invalid_data_length(self):
        with self.assertRaises(ValueError) as exc:
            self.pf.test_1.read(21)
        self.assertEqual(
            "invalid data length: 21 > 20", exc.exception.args[0]
        )
        with self.assertRaises(ValueError) as exc:
            self.pf.test_2.write([0] * 6)
        self.assertEqual(
            "invalid data length: 6 > 5", exc.exception.args[0]
        )

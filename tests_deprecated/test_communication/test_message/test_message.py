import unittest
from typing import get_args

import numpy as np

from ..utils import (
    validate_object,
    validate_fields,
)

from src.pyiak_instr.core import Code
from pyiak_instr_deprecation.communication import (
    Field,
    SingleField,
    StaticField,
    AddressField,
    DataField,
    IdField,
    OperationField,
    ResponseField,
    FieldSetter,
    MessageType,
    FieldMessage,
    SingleFieldMessage,
    StrongFieldMessage,
    MessageSetter,
    MessageContentError,
    NotConfiguredMessageError,
)


RESPONSE_CODES = {
    0: Code.OK,
    4: Code.WAIT
}


def test_common_methods(
        case: unittest.TestCase,
        res: FieldMessage,
        *,
        setter: MessageSetter,
        bytes_: bytes,
        length: int,
        string: str,
        init_mf_name: str = "std",
        init_splittable: bool = False,
        init_slice_length: int = 1024,
) -> None:
    with case.subTest(test="base init"):
        res_base = res.__class__()
        for name, ref in dict(
            mf_name=init_mf_name,
            splittable=init_splittable,
            slice_length=init_slice_length,
            src=None,
            dst=None,
        ).items():
            with case.subTest(name=name):
                case.assertEqual(ref, getattr(res_base, name))

    with case.subTest(test="get_instance"):
        instance = res.get_instance()
        case.assertIsInstance(instance, type(res))
        validate_object(
            case,
            instance,
            mf_name=res.mf_name,
            splittable=res.splittable,
            slice_length=res.slice_length,
        )

    with case.subTest(test="setter"):
        case.assertEqual(setter, res.get_setter())

    with case.subTest(test="bytes"):
        case.assertEqual(bytes_, bytes(res))

    with case.subTest(test="len"):
        case.assertEqual(length, len(res))

    with case.subTest(test="str"):
        case.assertEqual(string, str(res))

    with case.subTest(test="repr"):
        case.assertEqual(string, repr(res))

    # todo: get_instance


class TestFieldMessage(unittest.TestCase):

    maxDiff = None

    def test_base_init(self) -> None:
        validate_object(
            self,
            FieldMessage().configure(data=FieldSetter.data(expected=-1, fmt="B")),
            mf_name="std",
            splittable=False,
            slice_length=1024,
            src=None,
            dst=None,
            response_codes={},
            check_attrs=True,
            wo_attrs=["get", "has", "data"]
        )

    def test_common_methods(self) -> None:
        test_common_methods(
            self,
            FieldMessage(mf_name="test").configure(
                data=FieldSetter.data(expected=2, fmt="B")
            ).set(data=[0, 1]),
            setter=MessageSetter("field", "test", False, 1024),
            bytes_=b"\x00\x01",
            length=2,
            string="<FieldMessage(data=0 1), src=None, dst=None>",
            init_mf_name="std",
            init_splittable=False,
            init_slice_length=1024,
        )

    def test_configure_one_finite(self) -> None:
        msg = FieldMessage().configure(
            data=FieldSetter.data(expected=4, fmt="B")
        )
        validate_object(
            self,
            msg,
            mf_name="std",
            splittable=False,
            slice_length=1024,
            src=None,
            dst=None,
            response_codes={},
            wo_attrs=["get", "has", "data"],
            check_attrs=True,
        )
        validate_fields(
            self,
            msg,
            [DataField(
                "std",
                "data",
                start_byte=0,
                expected=4,
                fmt="B",
            )],
            wo_attrs=["parent"]
        )
        validate_object(
            self,
            msg.has,
            DataField=True,
            ResponseField=False,
            OperationField=False,
            infinite=False,
            DataLengthField=False,
            CrcField=False,
            AddressField=False,
            IdField=False,
            check_attrs=True,
        )

    def test_configure_one_infinite(self) -> None:
        msg = FieldMessage().configure(
            data=FieldSetter.data(expected=-1, fmt="B")
        )

        validate_fields(
            self,
            msg,
            [DataField(
                "std",
                "data",
                start_byte=0,
                expected=-1,
                fmt="B",
            )],
            wo_attrs=["parent"]
        )
        validate_object(
            self,
            msg.has,
            DataField=True,
            ResponseField=False,
            OperationField=False,
            infinite=True,
            DataLengthField=False,
            CrcField=False,
            AddressField=False,
            IdField=False,
            check_attrs=True,
        )

    def test_configure_first_infinite(self) -> None:
        msg = FieldMessage().configure(
            data=FieldSetter.data(expected=-1, fmt="B"),
            footer=FieldSetter.address(fmt=">H"),
            next=FieldSetter.static(fmt=">I", default=0x12345678)
        )

        fields = []
        f = DataField("std", "data", start_byte=0, expected=-1, fmt="B")
        f.stop_byte = -6
        fields.append(f)

        f = AddressField("std", "footer", start_byte=-6, fmt=">H")
        f.stop_byte = -4
        fields.append(f)

        f = StaticField(
            "std", "next", start_byte=-4, fmt=">I", default=0x12345678
        )
        f.stop_byte = None
        fields.append(f)

        validate_fields(self, msg, fields, wo_attrs=["parent"])
        validate_object(
            self,
            msg.has,
            DataField=True,
            ResponseField=False,
            OperationField=False,
            infinite=True,
            DataLengthField=False,
            CrcField=False,
            AddressField=True,
            IdField=False,
            check_attrs=True,
        )

    def test_configure_middle_infinite(self) -> None:
        msg = FieldMessage().configure(
            head=FieldSetter.operation(fmt=">H"),
            data=FieldSetter.data(expected=-1, fmt="B"),
            footer=FieldSetter.address(fmt=">I"),
        )

        fields = [OperationField("std", "head", start_byte=0, fmt=">H")]

        f = DataField("std", "data", start_byte=2, expected=-1, fmt="B")
        f.stop_byte = -4
        fields.append(f)

        f = AddressField("std", "footer", start_byte=-4, fmt=">I")
        f.stop_byte = None
        fields.append(f)

        validate_fields(self, msg, fields, wo_attrs=["parent"])
        validate_object(
            self,
            msg.has,
            DataField=True,
            ResponseField=False,
            OperationField=True,
            infinite=True,
            DataLengthField=False,
            CrcField=False,
            AddressField=True,
            IdField=False,
            check_attrs=True,
        )

    def test_configure_last_infinite(self) -> None:
        msg = FieldMessage().configure(
            head=FieldSetter.operation(fmt=">H"),
            next=FieldSetter.address(fmt=">I"),
            data=FieldSetter.data(expected=-1, fmt="B"),
        )

        fields = [OperationField("std", "head", start_byte=0, fmt=">H")]
        f = AddressField("std", "next", start_byte=2, fmt=">I")
        f.stop_byte = 6
        fields.append(f)

        f = DataField("std", "data", start_byte=6, expected=-1, fmt="B")
        f.stop_byte = None
        fields.append(f)

        validate_fields(self, msg, fields, wo_attrs=["parent"])
        validate_object(
            self,
            msg.has,
            DataField=True,
            ResponseField=False,
            OperationField=True,
            infinite=True,
            DataLengthField=False,
            CrcField=False,
            AddressField=True,
            IdField=False,
            check_attrs=True,
        )

    def test_configure_not_all_fields_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            FieldMessage().configure()
        self.assertEqual(
            "not all required fields were got: data are missing",
            exc.exception.args[0]
        )

    def test_configure_two_infinite_exc(self) -> None:
        with self.assertRaises(MessageContentError) as exc:
            FieldMessage().configure(
                data=FieldSetter.data(expected=-1, fmt="B"),
                data2=FieldSetter.base(expected=-1, fmt="B"),
            )
        self.assertEqual(
            "Error with data2 in FieldMessage: second infinite field",
            exc.exception.args[0]
        )

    def test_extract(self) -> None:
        msg = FieldMessage().configure(
            data=FieldSetter.data(expected=3, fmt=">H"),
            footer=FieldSetter.static(fmt=">I", default=0x1051291)
        ).extract(b"\x12\x23\x12\x89\x45\x12\x01\x05\x12\x91")

        df = DataField(
            mf_name="std",
            name="data",
            start_byte=0,
            expected=3,
            fmt=">H",
        )
        df.set(b"\x12\x23\x12\x89\x45\x12")
        validate_fields(
            self,
            msg,
            [
                df,
                StaticField(
                    mf_name="std",
                    name="footer",
                    start_byte=6,
                    fmt=">I",
                    default=0x1051291
                )
            ],
            wo_attrs=["parent"]
        )

    def test_extract_middle_infinite(self) -> None:
        msg = FieldMessage().configure(
            head=FieldSetter.static(fmt=">H", default=0x1234),
            data=FieldSetter.data(expected=-1, fmt=">H"),
            footer=FieldSetter.single(fmt="B")
        ).extract(b"\x12\x34\x12\x89\x45\x12\x01\x05\x12\xae\x91")

        fields = [
            StaticField(
                mf_name="std",
                name="head",
                start_byte=0,
                fmt=">H",
                default=0x1234,
            )
        ]

        f = DataField(
            mf_name="std",
            name="data",
            start_byte=2,
            expected=-1,
            fmt=">H",
        )
        f.stop_byte = -1
        f.set(b"\x12\x89\x45\x12\x01\x05\x12\xae")
        fields.append(f)

        f = SingleField(
            mf_name="std",
            name="footer",
            start_byte=0,
            fmt="B",
            default=b"",
        )
        f.start_byte = -1
        f.stop_byte = None
        f.set(0x91)
        fields.append(f)

        validate_fields(
            self,
            msg,
            fields,
            wo_attrs=["parent"]
        )

    def test_extract_not_configured_exc(self) -> None:
        with self.assertRaises(NotConfiguredMessageError) as exc:
            FieldMessage().extract(b"1234")
        self.assertEqual(
            "fields in FieldMessage instance not configured",
            exc.exception.args[0],
        )

    def test_get_field_by_type(self) -> None:
        msg = FieldMessage().configure(
            n1=FieldSetter.base(expected=1, fmt="b"),
            n2=FieldSetter.address(fmt="b"),
            data=FieldSetter.data(expected=1, fmt="b"),
            n3=FieldSetter.response(fmt="b", codes={}),
            n4=FieldSetter.base(expected=1, fmt="b"),
            n5=FieldSetter.address(fmt="b"),
            n6=FieldSetter.id_field(fmt="b"),
        )
        for name, cls in [
            ("n1", Field),
            ("n2", AddressField),
            ("data", DataField),
            ("n3", ResponseField),
            ("n1", Field),
            ("n2", AddressField),
            ("n6", IdField),
        ]:
            with self.subTest(field_class=cls.__name__):
                field = msg.get.field_by_type(cls)
                self.assertIsNotNone(field)
                self.assertEqual(name, field.name)

        with self.subTest(field_class="not exists"):
            with self.assertRaises(TypeError) as exc:
                self.assertIsNone(msg.get.OperationField)
            self.assertEqual(
                "there is no field with type OperationField",
                exc.exception.args[0]
            )

    def test_get_instance(self) -> None:
        ref = FieldMessage().configure(
            data=FieldSetter.data(expected=-1, fmt="B")
        ).set(data=255).set_src_dst(src="PC", dst="COM12")
        res = ref.get_instance()
        self.assertEqual(0, len(res))
        validate_object(
            self,
            res,
            mf_name="std",
            splittable=False,
            slice_length=1024,
            src=None,
            dst=None,
            response_codes={},
            check_attrs=True,
            wo_attrs=["get", "has", "data"]
        )
        validate_object(
            self,
            res.data,
            may_be_empty=True,
            fmt="B",
            finite=False,
            default=b"",
            name="data",
            expected=-1,
            stop_byte=None,
            words_count=0,
            content=b"",
            slice=slice(0, None),
            mf_name="std",
            start_byte=0,
            bytesize=1,
            check_attrs=True,
            wo_attrs=["parent"]
        )

    def test_set(self) -> None:
        msg = FieldMessage().configure(
            data=FieldSetter.data(expected=1, fmt="B")
        )
        msg.set(data=124)
        self.assertEqual(b"\x7c", msg.data.content)

    def test_set_with_crc(self) -> None:
        msg = FieldMessage().configure(
            data=FieldSetter.data(expected=1, fmt="B"),
            crc_name=FieldSetter.crc(fmt=">H"),
        )
        with self.subTest(test="auto crc"):
            msg.set(data=124)
            self.assertEqual(b"\xbf\x1b", msg["crc_name"].content)

        with self.subTest(test="manual crc"):
            with self.assertRaises(MessageContentError) as exc:
                msg.get_instance().set(data=123, crc_name=0xbf1c)
            self.assertEqual(
                "Error with crc_name in FieldMessage: "
                "invalid crc value, 'bf1c' != 'cffc'",
                exc.exception.args[0]
            )

    def test_set_with_data_length(self) -> None:
        msg = FieldMessage().configure(
            dlen=FieldSetter.dynamic_length(fmt="B"),
            data=FieldSetter.data(expected=1, fmt="I"),
        ).set(data=0x12345678)
        self.assertEqual(4, msg.get.DataLengthField[0])

    def test_set_not_configured_exc(self) -> None:
        with self.assertRaises(NotConfiguredMessageError) as exc:
            FieldMessage().set()
        self.assertEqual(
            "fields in FieldMessage instance not configured",
            exc.exception.args[0],
        )

    def test_src_dst(self) -> None:
        res = FieldMessage()
        self.assertTupleEqual((None, None), (res.src, res.dst))

        res.src = "PC"
        self.assertTupleEqual(("PC", None), (res.src, res.dst))

        res.dst = "COM1"
        self.assertTupleEqual(("PC", "COM1"), (res.src, res.dst))

        res.set_src_dst("COM1", "PC")
        self.assertTupleEqual(("COM1", "PC"), (res.src, res.dst))

        res.clear_src_dst()
        self.assertTupleEqual((None, None), (res.src, res.dst))

    def test_split(self):

        def get_mess(
                addr: int,
                oper: int,
                data_len: int,
                data=None,
                units=Code.WORDS,
        ):
            if data is None:
                data = b""
            return FieldMessage(
                splittable=True, slice_length=64
            ).configure(
                preamble=FieldSetter.static(fmt="B", default=0x12),
                address=FieldSetter.address(fmt="I"),
                data_length=FieldSetter.dynamic_length(fmt="B", units=units),
                test_field=FieldSetter.base(expected=1, fmt="B"),
                operation=FieldSetter.operation(fmt="B"),
                data=FieldSetter.data(expected=-1, fmt=">H")
            ).set(
                address=addr,
                data_length=data_len,
                test_field=0xff,
                operation=oper,
                data=data,
            )

        test_data = {
            0: (get_mess(0, 0, 34), (get_mess(0, 0, 34),)),
            1: (get_mess(0, 0, 64), (get_mess(0, 0, 64),)),
            2: (
                get_mess(0, 0, 128),
                (get_mess(0, 0, 64), get_mess(64, 0, 64)),
            ),
            3: (
                get_mess(0, 0, 127),
                (get_mess(0, 0, 64), get_mess(64, 0, 63)),
            ),
            4: (
                get_mess(0, 1, 34, range(34)),
                (get_mess(0, 1, 34, range(34)),),
            ),
            5: (
                get_mess(0, 1, 64, range(64)),
                (get_mess(0, 1, 64, range(64)),),
            ),
            6: (
                get_mess(0, 1, 128, range(128)),
                (
                    get_mess(0, 1, 64, range(64)),
                    get_mess(64, 1, 64, range(64, 128))
                ),
            ),
            7: (
                get_mess(0, 1, 127, range(127)),
                (
                    get_mess(0, 1, 64, range(64)),
                    get_mess(64, 1, 63, range(64, 127))
                ),
            ),
            8: (
                get_mess(0, 1, 126, range(63), units=Code.BYTES),
                (
                    get_mess(0, 1, 64, range(32), units=Code.BYTES),
                    get_mess(64, 1, 62, range(32, 63), units=Code.BYTES)
                ),
            )
        }

        for i_mess, (res, refs) in test_data.items():
            msgs = list(res.split())
            with self.subTest(test="split count", i_mess=i_mess):
                self.assertEqual(len(refs), len(msgs))
                for i_part, (msg, ref) in enumerate(zip(msgs, refs)):
                    with self.subTest(test="parts comparsion", i_part=i_part):
                        self.assertEqual(str(ref), str(msg))
                        self.assertEqual(ref.mf_name, msg.mf_name)

    def test_split_not_splittable(self) -> None:
        msg = FieldMessage(slice_length=2).configure(
            data=FieldSetter.data(expected=4, fmt="B")
        ).set(data=range(4))

        res = list(msg.split())
        self.assertEqual(1, len(res))
        self.assertIs(msg, res[0])

    def test_unpack(self):
        msg = FieldMessage(slice_length=2).configure(
            data=FieldSetter.data(expected=4, fmt="B")
        ).set(data=range(4))

        res = msg.unpack()
        self.assertIsInstance(res, np.ndarray)
        self.assertListEqual([0, 1, 2, 3], list(res))

    def test_validate_content(self) -> None:
        msg1 = FieldMessage().configure(
            preamble=FieldSetter.static(fmt="B", default=0xa5),
            response=FieldSetter.response(fmt="B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt="B"),
            operation=FieldSetter.operation(fmt="B"),
            data_length=FieldSetter.dynamic_length(fmt="B"),
            data=FieldSetter.data(expected=1, fmt="I"),
            crc=FieldSetter.crc(fmt="H"),
        ).set(
            response=0,
            address=0,
            operation=0,
            data_length=0
        )
        msg2 = FieldMessage().configure(
            address=FieldSetter.address(fmt="B"),
            operation=FieldSetter.operation(fmt="B"),
            data_length=FieldSetter.dynamic_length(
                fmt="B", units=Code.WORDS
            ),
            data=FieldSetter.data(expected=-1, fmt="B")
        ).set(address=0x12, operation=0x01, data_length=2)

        with self.subTest(name="may_be_empty"):
            msg2.set(data=[0x12, 0x56])
            self.assertListEqual([0x12, 0x56], list(msg2.data.unpack()))

            msg2.set(data=[])
            self.assertListEqual([], list(msg2.data.unpack()))

            with self.assertRaises(MessageContentError) as exc:
                msg2.set(operation=b"")
            self.assertEqual(
                "Error with operation in FieldMessage: field is empty",
                exc.exception.args[0],
            )

        with self.subTest(name="invalid data length"):
            with self.assertRaises(MessageContentError) as exc:
                msg2.set(address=0, operation="w", data_length=1, data=[0, 1])
            self.assertEqual(
                "Error with data_length in FieldMessage: invalid length",
                exc.exception.args[0]
            )

        with self.subTest(name="invalid crc"):
            with self.assertRaises(MessageContentError) as exc:
                msg1.set(crc=10)
            self.assertIn(
                "Error with crc in FieldMessage: invalid crc value, ",
                exc.exception.args[0]
            )

    def test_prop_response_codes(self) -> None:
        self.assertDictEqual(
            dict(
                pc_status=Code.OK, link_status=Code.WAIT
            ),
            FieldMessage().configure(
                data=FieldSetter.data(expected=1, fmt="B"),
                pc_status=FieldSetter.response(fmt="B", codes={1: Code.OK}),
                link_status=FieldSetter.response(
                    fmt="B", codes={3: Code.WAIT}
                )
            ).extract(b"\x05\x01\x03").response_codes
        )

        self.assertDictEqual(
            {},
            FieldMessage().configure(
                data=FieldSetter.data(expected=1, fmt="B")
            ).extract(b"1").response_codes
        )

    def test_magic_add(self):

        msg = FieldMessage().configure(
            preamble=FieldSetter.static(fmt="B", default=0x15),
            data=FieldSetter.data(expected=-1, fmt="B"),
        ).set(data=range(2))
        ref = [0, 1]

        with self.subTest(test="add bytes"):
            msg += b"\x23\x11"
            ref += [35, 17]
            self.assertListEqual(ref, list(msg.data))

        with self.subTest(test="add message"):
            msg += msg.get_instance().set(data=[12, 99])
            ref += [12, 99]
            self.assertListEqual(ref, list(msg.data))

        msg = FieldMessage().configure(
            data_length=FieldSetter.dynamic_length(fmt="B"),
            data=FieldSetter.data(expected=-1, fmt=">H"),
        ).set(data=range(2), data_length=4)
        ref = [0, 1]

        with self.subTest(test="update data_length"):
            self.assertEqual(4, msg["data_length"][0])
            msg += b"\x00\x12\x10\x44"
            ref += [0x12, 0x1044]
            self.assertListEqual(ref, list(msg.data))
            self.assertEqual(8, msg["data_length"][0])

    def test_magic_add_exc(self) -> None:
        msg = FieldMessage().configure(
            data=FieldSetter.data(expected=-1, fmt="B"),
        ).set()

        with self.subTest(test="add other mf_name"):
            with self.assertRaises(TypeError) as exc:
                msg += FieldMessage(mf_name="not_std")
            self.assertEqual(
                "messages have different formats: not_std != std",
                exc.exception.args[0]
            )

        with self.subTest(test="add invalid type"):
            with self.assertRaises(TypeError) as exc:
                msg += object()
            self.assertEqual(
                "<class 'object'> cannot be added to the message",
                exc.exception.args[0]
            )


class TestSingleFieldMessage(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            SingleFieldMessage(),
            mf_name="std",
            splittable=False,
            slice_length=1024,
            src=None,
            dst=None,
            check_attrs=True,
            wo_attrs=["get", "has", "data", "response_codes"]
        )

        validate_object(
            self,
            SingleFieldMessage([12, 3], ">H").data,
            finite=False,
            name="data",
            words_count=2,
            may_be_empty=True,
            content=b"\x00\x0c\x00\x03",
            default=b"",
            mf_name="std",
            start_byte=0,
            slice=slice(0, None),
            bytesize=2,
            fmt=">H",
            stop_byte=None,
            expected=-1,
            check_attrs=True,
            wo_attrs=["parent"]
        )

    def test_common_methods(self) -> None:
        test_common_methods(
            self,
            SingleFieldMessage(mf_name="test").configure(
                FieldSetter.data(expected=-1, fmt="B")
            ).set([0, 1]),
            setter=MessageSetter("single", "test", False, 1024),
            bytes_=b"\x00\x01",
            length=2,
            string="<SingleFieldMessage(data=0 1), src=None, dst=None>",
            init_mf_name="std",
            init_splittable=False,
            init_slice_length=1024,
        )

    def test_split(self) -> None:
        msg = SingleFieldMessage().set(
            bytearray(i % 256 for i in range(3, 2000))
        )
        for res in msg.split():
            with self.subTest(test="not splittable"):
                self.assertIs(msg, res)

        msg = SingleFieldMessage(splittable=True)\
            .set(bytearray(i % 256 for i in range(3, 2000)))
        for part, ref in zip(
                msg.split(),
                (
                    [i % 256 for i in range(3, 1027)],
                    [i % 256 for i in range(1027, 2000)],
                ),
        ):
            with self.subTest(test="splittable"):
                self.assertListEqual(ref, list(part.unpack()))

    def test_add(self) -> None:
        msg = SingleFieldMessage(data=b"\x01\xff\xa1")
        self.assertEqual([1, 255, 161], list(msg.unpack()))

        msg += b"\x17"
        self.assertEqual([1, 255, 161, 23], list(msg.unpack()))

        msg += SingleFieldMessage(b"\xbb")
        self.assertEqual([1, 255, 161, 23, 187], list(msg.unpack()))


class TestStrongFieldMessage(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            StrongFieldMessage().configure(
                address=FieldSetter.address(fmt="b"),
                operation=FieldSetter.operation(fmt="b"),
                data_length=FieldSetter.dynamic_length(fmt="b"),
                data=FieldSetter.data(expected=-1, fmt="b"),
            ),
            mf_name="std",
            splittable=False,
            slice_length=1024,
            src=None,
            dst=None,
            check_attrs=True,
            wo_attrs=[
                "get",
                "has",
                "data",
                "address",
                "operation",
                "data_length",
                "response_codes",
            ]
        )

    def test_common_methods(self) -> None:
        test_common_methods(
            self,
            StrongFieldMessage().configure(
                address=FieldSetter.address(fmt="b"),
                operation=FieldSetter.operation(fmt="b"),
                data_length=FieldSetter.dynamic_length(fmt="b"),
                data=FieldSetter.data(expected=-1, fmt="B"),
            ).set(address=1, operation=0, data_length=2, data=[5, 212]),
            setter=MessageSetter("strong"),
            bytes_=b"\x01\x00\x02\x05\xd4",
            length=5,
            string="<StrongFieldMessage(address=1, operation=0, "
                   "data_length=2, data=5 D4), src=None, dst=None>",
        )

    def test_required_fields(self) -> None:
        with self.assertRaises(ValueError) as exc:
            StrongFieldMessage().configure(
                n1=FieldSetter.address(fmt="b"),
            )
        self.assertIn(
            "not all required fields were got: ",
            exc.exception.args[0]
        )


class TestMessageSetter(unittest.TestCase):

    def test_init(self) -> None:
        res = MessageSetter()
        self.assertEqual("field", res.message_type)
        self.assertDictEqual(
            dict(mf_name="std", slice_length=1024, splittable=False),
            res.kwargs
        )

    def test_init_base(self) -> None:
        with self.assertRaises(ValueError) as exc:
            MessageSetter("base")
        self.assertEqual(
            "BaseMessage not supported by setter",
            exc.exception.args[0]
        )

    def test_invalid_message_type(self) -> None:
        with self.assertRaises(ValueError) as exc:
            MessageSetter("test")
        self.assertEqual(
            "invalid message type: 'test'", exc.exception.args[0]
        )

    def test_message(self) -> None:
        for msg_type, msg_class in MessageSetter.MESSAGE_TYPES.items():
            with self.subTest(message_type=msg_type):
                if msg_type == "base":
                    continue
                self.assertIsInstance(
                    MessageSetter(msg_type).message, msg_class
                )

    def test_message_class(self) -> None:
        for msg_type, msg_class in MessageSetter.MESSAGE_TYPES.items():
            with self.subTest(message_type=msg_type):
                if msg_type == "base":
                    continue

                res_class = MessageSetter(msg_type).message_class
                self.assertIs(res_class, msg_class)
                self.assertIn(
                    res_class,
                    get_args(MessageType),
                    "MessageType not supports %r" % msg_type
                )

    def test_init_kwargs(self) -> None:
        self.assertDictEqual(
            {
                "message_type": "field",
                "mf_name": "std",
                "slice_length": 1024,
                "splittable": False
            },
            MessageSetter("field").init_kwargs
        )

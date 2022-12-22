import unittest
from typing import Any, get_args

import numpy as np

from ..utils import (
    validate_object,
    compare_objects,
    validate_fields,
    compare_messages,
)

from pyinstr_iakoster.core import Code
from pyinstr_iakoster.communication import (
    ContentType,
    Field,
    SingleField,
    StaticField,
    AddressField,
    CrcField,
    DataField,
    DataLengthField,
    OperationField,
    ResponseField,
    FieldSetter,
    FieldType,
    FieldContentError,
    MessageType,
    BaseMessage,
    BytesMessage,
    FieldMessage,
    StrongFieldMessage,
    MessageSetter,
    MessageContentError,
    NotConfiguredMessageError,
    Register,
    RegisterMap,
    AsymmetricResponseField,
    MessageFormat,
    MessageFormatMap,
    PackageFormat,
)


RESPONSE_CODES = {
    0: Code.OK,
    4: Code.WAIT
}


def test_common_methods(
        case: unittest.TestCase,
        res: BaseMessage,
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


class TestBaseMessage(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            BaseMessage(),
            mf_name="std",
            splittable=False,
            slice_length=1024,
            src=None,
            dst=None,
            check_attrs=True,
        )

    def test_src_dst(self) -> None:
        res = BytesMessage()
        self.assertTupleEqual((None, None), (res.src, res.dst))

        res.src = "PC"
        self.assertTupleEqual(("PC", None), (res.src, res.dst))

        res.dst = "COM1"
        self.assertTupleEqual(("PC", "COM1"), (res.src, res.dst))

        res.set_src_dst("COM1", "PC")
        self.assertTupleEqual(("COM1", "PC"), (res.src, res.dst))

        res.clear_src_dst()
        self.assertTupleEqual((None, None), (res.src, res.dst))

    def test_get_instance(self) -> None:
        ref = BaseMessage(mf_name="test", slice_length=512)
        ref.set_src_dst("PC", "COM1")
        res = ref.get_instance()

        compare_objects(
            self, ref, res, attrs=["mf_name", "splittable", "slice_length"]
        )
        with self.subTest(test="src_dst"):
            self.assertTupleEqual((None, None), (res.src, res.dst))

    def test_get_setter(self) -> None:
        with self.assertRaises(ValueError) as exc:
            BaseMessage().get_setter()
        self.assertEqual(
            "BaseMessage not supported by setter", exc.exception.args[0]
        )

    def test_not_implemented(self) -> None:
        obj = BaseMessage()

        for method in [
            "set",
            "split",
            "in_bytes",
            "unpack",
            "_content_repr",
        ]:
            with self.subTest(method=method):
                with self.assertRaises(NotImplementedError):
                    getattr(obj, method)()

        with self.subTest(method="__add__"):
            with self.assertRaises(NotImplementedError):
                obj += 1

        with self.subTest(method="__getitem__"):
            with self.assertRaises(NotImplementedError):
                a = obj[""]

        with self.subTest(method="__iter__"):
            with self.assertRaises(NotImplementedError):
                for _ in obj:
                    ...

        for func in [len, bytes, repr, str]:
            with self.subTest(method=func.__name__):
                with self.assertRaises(NotImplementedError):
                    func(obj)


class AnotherMessage(StrongFieldMessage):

    def __init__(
            self,
            mf_name: str = "new_def",
            splittable: bool = False,
            slice_length: int = 1024
    ):
        StrongFieldMessage.__init__(
            self,
            mf_name=mf_name,
            splittable=splittable,
            slice_length=slice_length
        )


class TestBytesMessage(unittest.TestCase):

    def test_get_setter(self) -> None:
        compare_objects(
            self,
            MessageSetter("bytes"),
            BytesMessage().get_setter(),
            wo_attrs=["message"],
        )

    def test_init(self) -> None:
        validate_object(
            self,
            BytesMessage(),
            mf_name="std",
            splittable=False,
            slice_length=1024,
            src=None,
            dst=None,
            check_attrs=True,
        )

    def test_set(self) -> None:
        msg = BytesMessage(mf_name="test", content=b"\x01\x03\xff")
        self.assertEqual(b"\x01\x03\xff", msg.in_bytes())

        msg.set(bytearray(b"\x01\x03\xff"))
        self.assertListEqual([1, 3, 255], list(msg.unpack()))

    def test_split(self) -> None:
        msg = BytesMessage().set(
            bytearray(i % 256 for i in range(3, 2000))
        )
        for res in msg.split():
            with self.subTest(test="not splittable"):
                self.assertIs(msg, res)

        msg = BytesMessage(splittable=True)\
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
        msg = BytesMessage(content=b"\x01\xff\xa1")
        self.assertEqual([1, 255, 161], list(msg.unpack()))

        msg += b"\x17"
        self.assertEqual([1, 255, 161, 23], list(msg.unpack()))

        msg += BytesMessage(content=b"\xbb")
        self.assertEqual([1, 255, 161, 23, 187], list(msg.unpack()))

    def test_getitem(self) -> None:
        msg = BytesMessage(content=b"\x01\xff\xa1")
        self.assertEqual(255, msg[1])
        self.assertEqual(b"\xff\xa1", msg[1:])

    def test_iter(self) -> None:
        msg = BytesMessage(content=b"\x01\xff\xa1")
        for ref, res in zip([1, 255, 161], msg):
            self.assertEqual(ref, res)

    def test_common_methods(self) -> None:
        test_common_methods(
            self,
            BytesMessage(
                mf_name="test",
                content=b"\x01\xff\xa1",
            ),
            get_instance=BytesMessage(mf_name="test"),
            setter=MessageSetter("bytes", "test", False, 1024),
            bytes_=b"\x01\xff\xa1",
            length=3,
            string="<BytesMessage(01 ff a1), src=None, dst=None>",
            init_mf_name="std",
            init_splittable=False,
            init_slice_length=1024,
        )


class TestFieldMessage(unittest.TestCase):

    maxDiff = None

    def test_base_init(self) -> None:
        validate_object(
            self,
            FieldMessage(),
            mf_name="std",
            splittable=False,
            slice_length=1024,
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
            wo_attrs=["has", "data"],
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
        )
        for name, cls in [
            ("n1", Field),
            ("n2", AddressField),
            ("data", DataField),
            ("n3", ResponseField),
            ("n1", Field),
            ("n2", AddressField),
        ]:
            with self.subTest(field_class=cls.__name__):
                field = msg.get_field_by_type(cls)
                self.assertIsNotNone(field)
                self.assertEqual(name, field.name)

        with self.subTest(field_class="None"):
            self.assertIsNone(msg.get_field_by_type(OperationField))

    def test_get_instance(self) -> None:
        ref = FieldMessage().configure(
            data=FieldSetter.data(expected=-1, fmt="b")
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
            wo_attrs=["has", "data"]
        )
        validate_object(
            self,
            res.data,
            may_be_empty=True,
            fmt="b",
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

    def test_set_not_configured_exc(self) -> None:
        with self.assertRaises(NotConfiguredMessageError) as exc:
            FieldMessage().set()
        self.assertEqual(
            "fields in FieldMessage instance not configured",
            exc.exception.args[0],
        )

    def test_split(self):

        def get_mess(
                addr: int,
                oper: int,
                data_len: int,
                data=None,
                units=FieldSetter.WORDS,
        ):
            if data is None:
                data = b""
            return FieldMessage(
                splittable=True, slice_length=64
            ).configure(
                preamble=FieldSetter.static(fmt="B", default=0x12),
                address=FieldSetter.address(fmt="I"),
                data_length=FieldSetter.data_length(fmt="B", units=units),
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
                get_mess(0, 1, 126, range(63), units=FieldSetter.BYTES),
                (
                    get_mess(0, 1, 64, range(32), units=FieldSetter.BYTES),
                    get_mess(64, 1, 62, range(32, 63), units=FieldSetter.BYTES)
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
            data_length=FieldSetter.data_length(fmt="B"),
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
            data_length=FieldSetter.data_length(
                fmt="B", units=FieldSetter.WORDS
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
            data_length=FieldSetter.data_length(fmt="B"),
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
                msg += BytesMessage()
            self.assertIn(
                ".BytesMessage'> cannot be added to the message",
                exc.exception.args[0]
            )


class TestStrongFieldMessage(unittest.TestCase):

    def setUp(self) -> None:
        self.msg: StrongFieldMessage = StrongFieldMessage().configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=1, fmt=">I"),
            crc=FieldSetter.crc(fmt=">H"),
        )
        self.simple_msg: StrongFieldMessage = StrongFieldMessage().configure(
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(
                fmt=">B", units=FieldSetter.WORDS
            ),
            data=FieldSetter.data(expected=-1, fmt=">H")
        )

    def test_get_message_class(self) -> None:
        compare_objects(
            self,
            MessageSetter("strong_field"),
            StrongFieldMessage().get_setter(),
            wo_attrs=["message"]
        )

    def fill_content(self) -> bytes:
        content = b"\x1a\xa5\x00\x00\xaa\x01\x04\xff\xee\xdd\xcc\x39\x86"
        self.msg.extract(content)
        return content

    def test_init(self):
        msg = StrongFieldMessage()
        self.assertEqual("std", msg.mf_name)
        with self.assertRaises(KeyError) as exc:
            msg.data.unpack()
        self.assertEqual(
            "data", exc.exception.args[0]
        )

    def test_configure(self):
        msg = StrongFieldMessage(mf_name="not_def").configure(
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
                "address",
                start_byte=3,
                fmt=">H",
                parent=msg,
            ),
            operation=OperationField(
                "not_def",
                "operation",
                start_byte=5,
                fmt=">B",
                parent=msg,
            ),
            data_length=DataLengthField(
                "not_def",
                "data_length",
                start_byte=6,
                fmt=">B",
                parent=msg,
            ),
            data=DataField(
                "not_def",
                "data",
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
        msg = StrongFieldMessage(mf_name="inf").configure(
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
                "operation",
                start_byte=0,
                fmt=">B",
                parent=msg,
            ),
            data_length=DataLengthField(
                "inf",
                "data_length",
                start_byte=1,
                fmt=">B",
                parent=msg,
            ),
            data=DataField(
                "inf",
                "data",
                start_byte=2,
                expected=-1,
                fmt=">H",
                parent=msg,
            ),
            address=AddressField(
                "inf",
                "address",
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
            msg.in_bytes()
        )
        self.assertEqual(
            b"".join(f.content for f in fields.values()),
            msg.in_bytes()
        )

    def test_extract(self):
        msg: StrongFieldMessage = StrongFieldMessage(
            mf_name="not_def"
        ).configure(
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
        msg: StrongFieldMessage = StrongFieldMessage(
            mf_name="not_def"
        ).configure(
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
            msg.in_bytes()
        )

    def test_get_instance(self):
        msg: StrongFieldMessage = self.msg.get_instance()
        self.assertEqual("std", msg.mf_name)
        for ref, res in zip(self.msg, msg):
            self.assertEqual(ref.name, res.name)
            self.assertIn(str(res), ("1AA5", "0", "EMPTY"))

    def test_to_bytes(self):
        content = self.fill_content()
        self.assertEqual(content, self.msg.in_bytes())

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
            self.assertEqual("Error with operation in StrongFieldMessage: field is empty", exc.exception.args[0])

        with self.subTest(name="invalid data length"):
            with self.assertRaises(MessageContentError) as exc:
                self.simple_msg.set(
                    address=0, operation="w", data_length=1, data=[0, 1]
                )
            self.assertEqual(
                "Error with data_length in StrongFieldMessage: invalid length",
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
                "Error with crc in StrongFieldMessage: invalid crc value, ",
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
            ).in_bytes()
        )

    def test_split(self):

        def get_test_msg() -> StrongFieldMessage:
            new_msg = StrongFieldMessage(splittable=True, slice_length=64)
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
        msg = self.msg.get_instance()
        msg.set(
            preamble=0x1aa5,
            response=0,
            address=0x1234,
            operation=1,
            data_length=4,
            data=0x1234,
            crc=0xbd70,
        )
        self.msg += msg
        self.assertEqual(
            b"\xff\xee\xdd\xcc\x00\x00\x12\x34", self.msg.data.content
        )
        self.assertEqual(2, self.msg.data.expected)
        self.assertEqual(8, self.msg.data_length.unpack())

    def test_madic_add_errors(self):
        with self.assertRaises(TypeError) as exc:
            self.msg += self.msg.__class__(mf_name="new")
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
        msg_2 = msg_1.get_instance()
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
        msg: StrongFieldMessage = StrongFieldMessage(mf_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=1, fmt=">I"),
            crc=FieldSetter.base(expected=1, fmt=">H"),
        )
        msg.extract(b"\x1a\xa5\x32\x01\x55\x01\x04\xff\xff\xf1\xfe\xee\xdd")
        dlen = DataLengthField("def", "data_length", start_byte=0, fmt=">H")
        self.assertListEqual([], list(dlen.unpack()))
        dlen.set(dlen.calculate(msg.data))
        self.assertListEqual([4], list(dlen.unpack()))

    def test_operation_compare(self):
        msg: StrongFieldMessage = StrongFieldMessage(mf_name="not_def").configure(
            preamble=FieldSetter.static(fmt=">H", default=0x1aa5),
            response=FieldSetter.response(fmt=">B", codes=RESPONSE_CODES),
            address=FieldSetter.address(fmt=">H"),
            operation=FieldSetter.operation(fmt=">B"),
            data_length=FieldSetter.data_length(fmt=">B"),
            data=FieldSetter.data(expected=1, fmt=">I"),
            crc=FieldSetter.base(expected=1, fmt=">H"),
        )
        msg.extract(b"\x1a\xa5\x32\x01\x55\x01\x04\xff\xff\xf1\xfe\xee\xdd")
        oper = OperationField("def", "operation", start_byte=0, fmt=">H")
        self.assertFalse(oper.compare(msg))
        oper.set("w")
        self.assertTrue(oper.compare(msg))
        oper.set("r")
        self.assertFalse(oper.compare(msg))


class TestMessageSetter(unittest.TestCase):

    def test_init(self) -> None:
        res = MessageSetter()
        self.assertEqual("bytes", res.message_type)
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

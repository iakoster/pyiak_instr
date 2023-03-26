# import unittest
#
# from src.pyiak_instr.core import Code
# from src.pyiak_instr.communication.message import (
#     MessageFieldStruct,
#     SingleMessageFieldStruct,
#     StaticMessageFieldStruct,
#     AddressMessageFieldStruct,
#     CrcMessageFieldStruct,
#     DataMessageFieldStruct,
#     DataLengthMessageFieldStruct,
#     IdMessageFieldStruct,
#     OperationMessageFieldStruct,
#     ResponseMessageFieldStruct,
#     MessageFieldPattern,
# )
#
# from ....utils import validate_object
#
#
# class TestMessageFieldParameters(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             MessageFieldStruct(
#                 bytes_expected=20,
#                 fmt=Code.I64,
#                 start=10,
#             ),
#             bytes_expected=160,
#             default=b"",
#             expected=20,
#             fmt=Code.I64,
#             infinite=False,
#             order=Code.BIG_ENDIAN,
#             slice=slice(10, 170),
#             start=10,
#             stop=170,
#             word_size=8,
#         )
#
#
# class TestSingleMessageFieldParameters(unittest.TestCase):
#
#     def test_init_exc(self) -> None:
#         with self.assertRaises(ValueError) as exc:
#             SingleMessageFieldStruct(
#                 fmt=Code.U8,
#                 start=0,
#                 expected=2,
#             )
#         self.assertEqual(
#             "single field should expect one word", exc.exception.args[0]
#         )
#
#     def test_validate(self) -> None:
#         datas = (
#             (b"", False),
#             (b"\x00", False),
#             (b"\x00" * 2, True),
#             (b"\x00" * 3, False),
#         )
#         obj = SingleMessageFieldStruct(
#             fmt=Code.U16,
#             start=0,
#         )
#
#         for i, (data, ref) in enumerate(datas):
#             with self.subTest(test=i):
#                 self.assertEqual(ref, obj.validate(data))
#
#
# class TestStaticMessageFieldParameters(unittest.TestCase):
#
#     def test_init_exc(self) -> None:
#         with self.assertRaises(ValueError) as exc:
#             StaticMessageFieldStruct(
#                 fmt=Code.U8,
#                 start=0,
#             )
#         self.assertEqual("default value not specified", exc.exception.args[0])
#
#     def test_validate(self) -> None:
#         obj = StaticMessageFieldStruct(
#             fmt=Code.U16,
#             start=0,
#             default=b"42"
#         )
#         for i, (data, ref) in enumerate((
#             (b"41", False),
#             (b"42", True),
#         )):
#             self.assertEqual(ref, obj.validate(data))
#
#
# class TestAddressMessageFieldParameters(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             AddressMessageFieldStruct(
#                 fmt=Code.I64,
#                 start=10,
#             ),
#             bytes_expected=8,
#             default=b"",
#             expected=1,
#             fmt=Code.I64,
#             infinite=False,
#             order=Code.BIG_ENDIAN,
#             slice=slice(10, 18),
#             start=10,
#             stop=18,
#             word_size=8,
#         )
#
#
# class TestCrcMessageFieldParameters(unittest.TestCase):
#
#     def test_init_exc(self) -> None:
#         with self.assertRaises(ValueError) as exc:
#             CrcMessageFieldStruct(fmt=Code.U8, start=0, algorithm_name="None")
#         self.assertEqual("invalid algorithm: 'None'", exc.exception.args[0])
#
#     def test_algorithm(self) -> None:
#         obj = CrcMessageFieldStruct(fmt=Code.U16, start=0)
#
#         for i, (data, ref) in enumerate((
#             (b"\x10\x01\x20\x04", 0x6af5),
#             (bytes(range(15)), 0x9b92),
#             (bytes(i % 256 for i in range(1500)), 0x9243),
#             (b"\x01\x00\x00\x00\x00\x00", 0x45a0),
#         )):
#             with self.subTest(test=i):
#                 self.assertEqual(ref, obj.algorithm(data))
#
#
# class TestDataMessageFieldParameters(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             DataMessageFieldStruct(
#                 fmt=Code.I64,
#                 start=10,
#                 expected=-1,
#             ),
#             bytes_expected=0,
#             default=b"",
#             expected=0,
#             fmt=Code.I64,
#             infinite=True,
#             order=Code.BIG_ENDIAN,
#             slice=slice(10, None),
#             start=10,
#             stop=None,
#             word_size=8,
#         )
#
#
# class TestDataLengthMessageFieldParameters(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             DataLengthMessageFieldStruct(
#                 fmt=Code.U16,
#                 start=10,
#             ),
#             additive=0,
#             behaviour=Code.ACTUAL,
#             bytes_expected=2,
#             default=b"",
#             expected=1,
#             fmt=Code.U16,
#             infinite=False,
#             order=Code.BIG_ENDIAN,
#             slice=slice(10, 12),
#             start=10,
#             stop=12,
#             units=Code.BYTES,
#             word_size=2,
#         )
#
#     def test_init_exc(self) -> None:
#
#         for i, (msg, kw) in enumerate((
#             (
#                 "additive number must be positive integer, got -1",
#                 {"additive": -1},
#             ), (
#                 "invalid behaviour: <Code.WRITE: 1538>",
#                 {"behaviour": Code.WRITE},
#             ), (
#                 "invalid units: <Code.ACTUAL: 1536>",
#                 {"units": Code.ACTUAL},
#             ),
#         )):
#             with self.subTest(test=i):
#                 with self.assertRaises(ValueError) as exc:
#                     DataLengthMessageFieldStruct(fmt=Code.U8, start=0, **kw)
#                 self.assertEqual(msg, exc.exception.args[0])
#
#
# class TestIdMessageFieldParameters(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             IdMessageFieldStruct(
#                 fmt=Code.U16,
#                 start=10,
#             ),
#             bytes_expected=2,
#             default=b"",
#             expected=1,
#             fmt=Code.U16,
#             infinite=False,
#             order=Code.BIG_ENDIAN,
#             slice=slice(10, 12),
#             start=10,
#             stop=12,
#             word_size=2,
#         )
#
#
# class TestOperationMessageFieldParameters(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             OperationMessageFieldStruct(
#                 fmt=Code.U16,
#                 start=10,
#             ),
#             bytes_expected=2,
#             default=b"",
#             descriptions={Code.READ: 0, Code.WRITE: 1},
#             expected=1,
#             fmt=Code.U16,
#             infinite=False,
#             order=Code.BIG_ENDIAN,
#             slice=slice(10, 12),
#             start=10,
#             stop=12,
#             word_size=2,
#         )
#
#
# class TestResponseMessageFieldParameters(unittest.TestCase):
#
#     def test_init(self) -> None:
#         validate_object(
#             self,
#             ResponseMessageFieldStruct(
#                 fmt=Code.U16,
#                 start=10,
#                 codes={0: Code.OK, 1: Code.ERROR},
#             ),
#             bytes_expected=2,
#             codes={0: Code.OK, 1: Code.ERROR},
#             default=b"",
#             default_code=Code.UNDEFINED,
#             expected=1,
#             fmt=Code.U16,
#             infinite=False,
#             order=Code.BIG_ENDIAN,
#             slice=slice(10, 12),
#             start=10,
#             stop=12,
#             word_size=2,
#         )
#
#
# class TestMessageFieldPattern(unittest.TestCase):
#
#     def test_get(self) -> None:
#         data = dict(
#             basic=(
#                 MessageFieldPattern.basic(fmt=Code.U8, expected=0),
#                 MessageFieldStruct,
#             ),
#             single=(
#                 MessageFieldPattern.single(fmt=Code.U8), SingleMessageFieldStruct,
#             ),
#             static=(
#                 MessageFieldPattern.static(
#                     fmt=Code.U32, default=b"\x00\x01\x02\x03"
#                 ),
#                 StaticMessageFieldStruct,
#             ),
#             address=(
#                 MessageFieldPattern.address(fmt=Code.U8),
#                 AddressMessageFieldStruct,
#             ),
#             crc=(
#                 MessageFieldPattern.crc(fmt=Code.U16), CrcMessageFieldStruct,
#             ),
#             data=(
#                 MessageFieldPattern.data(fmt=Code.U64), DataMessageFieldStruct,
#             ),
#             data_length=(
#                 MessageFieldPattern.data_length(fmt=Code.I32),
#                 DataLengthMessageFieldStruct,
#             ),
#             id=(
#                 MessageFieldPattern.id(fmt=Code.U8), IdMessageFieldStruct,
#             ),
#             operation=(
#                 MessageFieldPattern.operation(fmt=Code.U16), OperationMessageFieldStruct,
#             ),
#             response=(
#                 MessageFieldPattern.response(fmt=Code.F16), ResponseMessageFieldStruct,
#             )
#         )
#
#         self.assertSetEqual(set(data), set(MessageFieldPattern._target_options))
#         for type_name, (pattern, field_type) in data.items():
#             pattern: MessageFieldPattern
#             with self.subTest(test=type_name):
#                 self.assertEqual(type_name, pattern.typename)
#                 res = pattern.get(start=0)
#
#                 for key, val in pattern.__init_kwargs__().items():
#                     if key == "typename":
#                         continue
#                     with self.subTest(parameter=key):
#                         self.assertEqual(val, getattr(res, key))

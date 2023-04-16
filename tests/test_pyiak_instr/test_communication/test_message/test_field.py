
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

# import unittest
# from typing import Any
#
# from src.pyiak_instr.core import Code
# from src.pyiak_instr.exceptions import NotAmongTheOptions, ContentError
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
#     MessageFieldStructUnionT,
#     MessageFieldPattern,
# )
#
# from ....utils import validate_object
#
#
# class TestMessageFieldPattern(unittest.TestCase):
#
#     def test_get(self) -> None:
#         self.assertIsInstance(
#             MessageFieldPattern(
#                 "basic", start=0, bytes_expected=0
#             ).get(fmt=Code.U8),
#             MessageFieldStruct,
#         )
#
#     def test_basic(self) -> None:
#         self._validate(
#             MessageFieldStruct,
#             MessageFieldPattern.basic(fmt=Code.U16).get(start=10),
#             0,
#             slice(10, None),
#             2,
#         )
#
#     def test_single(self) -> None:
#         self._validate(
#             SingleMessageFieldStruct,
#             MessageFieldPattern.single(fmt=Code.U24).get(start=3),
#             3,
#             slice(3, 6),
#             3,
#         )
#
#     def test_static(self) -> None:
#         self._validate(
#             StaticMessageFieldStruct,
#             MessageFieldPattern.static(
#                 fmt=Code.U40, default=b"\x00\x02\x03\x04\x05"
#             ).get(start=2),
#             5,
#             slice(2, 7),
#             5,
#             default=b"\x00\x02\x03\x04\x05",
#         )
#
#     def test_address(self) -> None:
#         self._validate(
#             AddressMessageFieldStruct,
#             MessageFieldPattern.address(fmt=Code.U8).get(start=0),
#             1,
#             slice(0, 1),
#             1,
#             behaviour=Code.DMA,
#         )
#
#     def test_crc(self) -> None:
#         self._validate(
#             CrcMessageFieldStruct,
#             MessageFieldPattern.crc(fmt=Code.U16).get(start=-2),
#             2,
#             slice(-2, None),
#             2,
#             poly=0x1021,
#             init=0,
#             wo_fields=set(),
#         )
#
#     def test_data(self) -> None:
#         self._validate(
#             DataMessageFieldStruct,
#             MessageFieldPattern.data(
#                 fmt=Code.U8, bytes_expected=1
#             ).get(start=-3),
#             1,
#             slice(-3, -2),
#             1,
#         )
#
#     def test_data_length(self) -> None:
#         self._validate(
#             DataLengthMessageFieldStruct,
#             MessageFieldPattern.data_length(fmt=Code.U8).get(start=4),
#             1,
#             slice(4, 5),
#             1,
#             behaviour=Code.ACTUAL,
#             units=Code.BYTES,
#             additive=0,
#         )
#
#     def test_id(self) -> None:
#         self._validate(
#             IdMessageFieldStruct,
#             MessageFieldPattern.id_(fmt=Code.U48).get(start=5),
#             6,
#             slice(5, 11),
#             6,
#         )
#
#     def test_operation(self) -> None:
#         self._validate(
#             OperationMessageFieldStruct,
#             MessageFieldPattern.operation(fmt=Code.U16).get(start=1),
#             2,
#             slice(1, 3),
#             2,
#             descs={0: Code.READ, 1: Code.WRITE},
#             descs_r={Code.READ: 0, Code.WRITE: 1},
#         )
#
#     def test_response(self) -> None:
#         self._validate(
#             ResponseMessageFieldStruct,
#             MessageFieldPattern.response(fmt=Code.U8).get(start=0),
#             1,
#             slice(0, 1),
#             1,
#             descs={},
#             descs_r={},
#         )
#
#     def _validate(
#             self,
#             ref_class: type[MessageFieldStructUnionT],
#             ref: MessageFieldStructUnionT,
#             bytes_expected: int,
#             slice_: slice,
#             word_bytesize: int,
#             default=b"",
#             order: Code = Code.BIG_ENDIAN,
#             **kwargs: Any,
#     ) -> None:
#         self.assertIsInstance(ref, ref_class)
#         kw = dict(
#             bytes_expected=bytes_expected,
#             default=default,
#             order=order,
#             slice_=slice_,
#             word_bytesize=word_bytesize,
#         )
#         kw.update(**kwargs)
#
#         validate_object(
#             self,
#             ref,
#             **{k: kw[k] for k in sorted(kw)},
#             wo_attrs=[
#                 "fmt",
#                 "has_default",
#                 "is_dynamic",
#                 "start",
#                 "stop",
#                 "words_expected",
#                 "units"
#             ],
#         )

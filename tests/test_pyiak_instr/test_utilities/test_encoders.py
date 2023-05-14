import unittest
from itertools import chain

import numpy as np
from numpy.testing import assert_array_almost_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import CodeNotAllowed
from src.pyiak_instr.utilities import BytesEncoder, StringEncoder

from ...utils import validate_object


def _get_value_instance(
    code: Code,
    length: int,
    encode_raw: list[bytes] | None = None,
    decode: list[int] | None = None,
):
    if encode_raw is None:
        encode_raw = [b"\x02", b"\x03"]
    if decode is None:
        decode = [2, 3]

    encode = b""
    for val in encode_raw:
        encode += b"\x00" * (length - len(val)) + val
    return code, decode, encode


class TestBytesEncoder(unittest.TestCase):

    DATA = dict(
        empty=(Code.U24, [], b"", Code.BIG_ENDIAN),
        u8=(Code.U8, [2, 3], b"\x02\x03", Code.BIG_ENDIAN),
        u24=(
            Code.U24,
            [0xFF1234, 0x2307],
            b"\xff\x12\x34\x00\x23\x07",
            Code.BIG_ENDIAN,
        ),
        u24_little=(
            Code.U24,
            [0x3412FF, 0x72300],
            b"\xff\x12\x34\x00\x23\x07",
            Code.LITTLE_ENDIAN,
        ),
        u24_single=(
            Code.U24,
            0x123456,
            b"\x12\x34\x56",
            Code.BIG_ENDIAN,
        ),
        i8=(Code.I8, [-127, -37], b"\x81\xdb", Code.BIG_ENDIAN),
        i32=(
            Code.I32,
            [-0xfabdc, -0xea],
            b"\xff\xf0\x54\x24\xff\xff\xff\x16",
            Code.BIG_ENDIAN,
        ),
        f16=(
            Code.F16,
            [1.244140625],
            b"\x3C\xFA",
            Code.BIG_ENDIAN,
        ),
        f32=(
            Code.F32,
            [6547.525390625],
            b"\x45\xCC\x9C\x34",
            Code.BIG_ENDIAN,
        ),
        f64=(
            Code.F64,
            [3.141592653589793],
            b"\x40\x09\x21\xFB\x54\x44\x2D\x18",
            Code.BIG_ENDIAN,
        ),
    )

    def test_init(self) -> None:
        validate_object(
            self,
            BytesEncoder(),
            value_size=1,
        )

    def test_decode(self) -> None:
        for name, (fmt, decoded, encoded, order) in self.DATA.items():
            with self.subTest(test=name, fmt=repr(fmt)):
                assert_array_almost_equal(
                    decoded,
                    BytesEncoder(fmt=fmt, order=order).decode(encoded),
                )

    def test_encode(self) -> None:
        for name, (fmt, decoded, encoded, order) in self.DATA.items():
            with self.subTest(test=name, fmt=repr(fmt)):
                self.assertEqual(
                    encoded,
                    BytesEncoder(fmt=fmt, order=order).encode(decoded),
                )

        with self.subTest(test="bytes", fmt=repr(Code.F64)):
            self.assertEqual(
                b"abc", BytesEncoder(fmt=Code.F64).encode(b"abc")
            )

    def test_verify_fmt_order_exc(self) -> None:
        with self.assertRaises(CodeNotAllowed) as exc:
            BytesEncoder().verify_fmt_order(Code.NONE, Code.NONE)
        self.assertEqual(
            "code option not allowed, got <Code.NONE: 0>",
            exc.exception.args[0],
        )

        with self.assertRaises(CodeNotAllowed) as exc:
            BytesEncoder().verify_fmt_order(Code.U8, Code.NONE)
        self.assertEqual(
            "code option not allowed, got <Code.NONE: 0>",
            exc.exception.args[0],
        )

    def test_value_size(self) -> None:
        cases = (
            (Code.U16, 2),
            (Code.U40, 5),
            (Code.I56, 7),
            (Code.F32, 4),
        )
        for code, bytesize in cases:
            with self.subTest(code=repr(code)):
                self.assertEqual(bytesize, BytesEncoder(fmt=code).value_size)

    def test_value_size_exc(self) -> None:
        with self.assertRaises(AssertionError) as exc:
            obj = BytesEncoder()
            obj._fmt = Code.FLOAT
            _ = obj.value_size
        self.assertEqual(
            "invalid value format: <Code.FLOAT: 260>", exc.exception.args[0]
        )

class TestStringEncoder(unittest.TestCase):

    DATA = dict(
        empty_str=("", ""),     # if empty string is a last parameter
                                # (e.g. dict) -> raise error
        letter=("a", "a"),
        word=("lol", "lol"),
        sentence=("lol kek!!!", "lol kek!!!"),
        bytes=("\\bts(1,3,255,50,120)", b"\x01\x03\xff\x32\x78"),
        int=("1", 1),
        int_neg=("-1", -1),
        int_str=("\\str(3)", "3"),
        float=("5.4321", 5.4321),
        float_neg=("-5.4321", -5.4321),
        float_str=("\\str(3.7)", "3.7"),
        efloat_small=("5.4321e-99", 5.4321e-99),
        efloat_small_neg=("-5.4321e-99", -5.4321e-99),
        efloat_str=("\\str(4.321e-107)", "4.321e-107"),
        efloat_large=("5.4321e+99", 5.4321e+99),
        code=("\\cod(1)", Code.OK),
        true=("True", True),
        true_str=("\\str(True)", "True"),
        false=("False", False),
        false_str=("\\str(False)", "False"),
        none=("None", None),
        none_str=("\\str(None)", "None"),
        list_empty=("\\lst()", []),
        list_diff=(
            "\\lst(a,1,1.1,1.1e-99,1.1e+99)", ["a", 1, 1.1, 1.1e-99, 1.1e+99]
        ),
        list_str_int=("\\lst(\\str(1))", ["1"]),
        tuple_empty=("\\tpl()", ()),
        tuple_diff=(
            "\\tpl(a,1,1.1,1.1e-99,1.1e+99)", ("a", 1, 1.1, 1.1e-99, 1.1e+99)
        ),
        dict_empty=("\\dct()", {}),
        dict_diff=(
            "\\dct(a,1,1.1,1.1e-99,1.1e+99,1.1e+178)",
            {"a": 1, 1.1: 1.1e-99, 1.1e+99: np.float64(1.1e+178)}
        ),
        set_empty=("\\set()", set()),
        set_int=("\\set(1,2,3)", {1, 2, 3}),
        array_empty=("\\npa()", np.array([])),
        array_int=("\\npa(1,2)", np.array([1, 2])),
        array_float=("\\npa(1.1,2.2)", np.array([1.1, 2.2])),
        array_efloat=("\\npa(1.1e-99,2.2e+99)", np.array([1.1e-99, 2.2e+99]))
    )
    DATA_CHAIN = dict(
        ch1=("\\dct(a,\\dct(1,1.1))", {"a": {1: 1.1}}),
        ch2=("\\dct(\\str(0),\\lst(),2,\\set(1,2))", {"0": [], 2: {1, 2}}),
        ch3=(
            "\\lst(0,-1.1,\\npa(),\\npa(2),\\dct())",
            [0, -1.1, np.array([]), np.array([2]), {}],
        ),
        ch4=(
            "\\npa("
            "\\npa(\\npa(0,-1),\\npa(2,3)),"
            "\\npa(\\npa(4,5),\\npa(6,7))"
            ")",
            np.array([[[0, -1], [2, 3]], [[4, 5], [6, 7]]])
        )
    )

    def test_init(self) -> None:
        validate_object(
            self,
            StringEncoder(),
            value_size=-1
        )

    def test_decode(self) -> None:
        for name, (src, ref) in chain(
                self.DATA.items(), self.DATA_CHAIN.items()
        ):
            with self.subTest(name=name, ref=ref):
                res = StringEncoder().decode(src)

                if name in ("ch3",):
                    self.assertEqual(str(res), str(ref))
                elif isinstance(ref, np.ndarray):
                    np.testing.assert_array_equal(ref, res)
                else:
                    self.assertEqual(ref, res)

    def test_decode_single(self) -> None:
        self.assertEqual(
            [1, 2, (3,)], StringEncoder().decode(r"\lst(1,2,\tpl(3))")
        )

    def test_encode(self):

        for name, (ref, src) in chain(
                self.DATA.items(), self.DATA_CHAIN.items()
        ):
            with self.subTest(name=name, ref=ref):
                self.assertEqual(ref, StringEncoder().encode(src))

    def test_encode_single(self) -> None:
        self.assertEqual(
            r"\lst(1,2,\tpl(3))", StringEncoder().encode([1, 2, (3,)])
        )

    def test_encode_invalid(self) -> None:
        self.assertEqual(
            "\\tpl\t1,2,3", StringEncoder().encode("\\tpl\t1,2,3")
        )

    def test_encode_code_support(self) -> None:
        self.assertEqual(
            "\cod(1)", StringEncoder().encode(Code.OK)
        )

    def test_decorate(self) -> None:
        self.assertEqual(
            "\\dct(1,2,3,4)", StringEncoder()._decorate(Code.DICT, "1,2,3,4")
        )

    def test_determine_type(self) -> None:
        for string, ref in (
            ("", Code.STRING),
            (r"\tpl(1,2)", Code.STRING),
            ("112", Code.INT),
            ("-15675", Code.INT),
            ("1.1", Code.FLOAT),
            ("-1.1e+10", Code.FLOAT),
            ("True", Code.BOOL),
            ("False", Code.BOOL),
            ("None", Code.NONE),
            ("text", Code.STRING),
        ):
            with self.subTest(string=string, ref=repr(ref)):
                res = StringEncoder()._determine_type(string)
                self.assertEqual(ref, res)

    def test_get_data_border(self) -> None:
        func = StringEncoder()._get_data_border
        self.assertTupleEqual((4, 8), func("\\tpl(1,2)"))
        self.assertTupleEqual((4, 8), func("\\lst(1,2),3,4)"))

    def test_get_data_border_exc(self) -> None:
        for string, msg in (
            ("", "string does not have SOD"),
            ("(", "SOD not closed in '('"),
        ):
            with self.subTest(string=string, msg=msg):
                with self.assertRaises(ValueError) as exc:
                    StringEncoder()._get_data_border(string)
                self.assertEqual(msg, exc.exception.args[0])

    def test_iter(self) -> None:
        self.assertListEqual(
            ["1", "2", "\\tpl(3,4)", "gg"],
            [*StringEncoder()._iter("1,2,\\tpl(3,4),gg")]
        )

    def test_read(self) -> None:
        self.assertTupleEqual(
            (Code.TUPLE, "1,2"), StringEncoder()._read("\\tpl(1,2)")
        )

    def test_split(self) -> None:
        for string, ref in (
            (r"\lol(1,2,3)", ("lol", "1,2,3")),
        ):
            self.assertTupleEqual(ref, StringEncoder()._split(string))


import unittest
from itertools import chain

import numpy as np
from numpy.testing import assert_array_almost_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import CodeNotAllowed
from src.pyiak_instr.codecs import StringCodec

from tests.utils import validate_object


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
            StringCodec(),
        )

    def test_decode(self) -> None:
        for name, (src, ref) in chain(
                self.DATA.items(), self.DATA_CHAIN.items()
        ):
            with self.subTest(name=name, ref=ref):
                res = StringCodec().decode(src)

                if name in ("ch3",):
                    self.assertEqual(str(res), str(ref))
                elif isinstance(ref, np.ndarray):
                    np.testing.assert_array_equal(ref, res)
                else:
                    self.assertEqual(ref, res)

    def test_decode_single(self) -> None:
        self.assertEqual(
            [1, 2, (3,)], StringCodec().decode(r"\lst(1,2,\tpl(3))")
        )

    def test_encode(self):

        for name, (ref, src) in chain(
                self.DATA.items(), self.DATA_CHAIN.items()
        ):
            with self.subTest(name=name, ref=ref):
                self.assertEqual(ref, StringCodec().encode(src))

    def test_encode_single(self) -> None:
        self.assertEqual(
            r"\lst(1,2,\tpl(3))", StringCodec().encode([1, 2, (3,)])
        )

    def test_encode_invalid(self) -> None:
        self.assertEqual(
            "\\tpl\t1,2,3", StringCodec().encode("\\tpl\t1,2,3")
        )

    def test_encode_code_support(self) -> None:
        self.assertEqual(
            "\cod(1)", StringCodec().encode(Code.OK)
        )

    def test_decorate(self) -> None:
        self.assertEqual(
            "\\dct(1,2,3,4)", StringCodec()._decorate(Code.DICT, "1,2,3,4")
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
                res = StringCodec()._determine_type(string)
                self.assertEqual(ref, res)

    def test_get_data_border(self) -> None:
        func = StringCodec()._get_data_border
        self.assertTupleEqual((4, 8), func("\\tpl(1,2)"))
        self.assertTupleEqual((4, 8), func("\\lst(1,2),3,4)"))

    def test_get_data_border_exc(self) -> None:
        for string, msg in (
            ("", "string does not have SOD"),
            ("(", "SOD not closed in '('"),
        ):
            with self.subTest(string=string, msg=msg):
                with self.assertRaises(ValueError) as exc:
                    StringCodec()._get_data_border(string)
                self.assertEqual(msg, exc.exception.args[0])

    def test_iter(self) -> None:
        self.assertListEqual(
            ["1", "2", "\\tpl(3,4)", "gg"],
            [*StringCodec()._iter("1,2,\\tpl(3,4),gg")]
        )

    def test_read(self) -> None:
        self.assertTupleEqual(
            (Code.TUPLE, "1,2"), StringCodec()._read("\\tpl(1,2)")
        )

    def test_split(self) -> None:
        for string, ref in (
            (r"\lol(1,2,3)", ("lol", "1,2,3")),
        ):
            self.assertTupleEqual(ref, StringCodec()._split(string))


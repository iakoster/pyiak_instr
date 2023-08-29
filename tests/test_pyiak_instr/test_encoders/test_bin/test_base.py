import unittest
from itertools import chain

import numpy as np
from numpy.testing import assert_array_almost_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotAmongTheOptions
from src.pyiak_instr.encoders.bin import (
    BytesIntDecoder,
    BytesIntEncoder,
    BytesFloatDecoder,
    BytesFloatEncoder,
    get_bytes_transformers,
)

from tests.utils import validate_object


FOR_INT = dict(
        empty=([], b"", Code.U24, Code.BIG_ENDIAN),
        u8=([2, 3], b"\x02\x03", Code.U8, Code.BIG_ENDIAN),
        u24=(
            [0xFF1234, 0x2307],
            b"\xff\x12\x34\x00\x23\x07",
            Code.U24,
            Code.BIG_ENDIAN,
        ),
        u24_little=(
            [0x3412FF, 0x72300],
            b"\xff\x12\x34\x00\x23\x07",
            Code.U24,
            Code.LITTLE_ENDIAN,
        ),
        u24_single=(
            0x123456,
            b"\x12\x34\x56",
            Code.U24,
            Code.BIG_ENDIAN,
        ),
        i8=([-127, -37], b"\x81\xdb", Code.I8, Code.BIG_ENDIAN),
        i32=(
            [-0xfabdc, -0xea],
            b"\xff\xf0\x54\x24\xff\xff\xff\x16",
            Code.I32,
            Code.BIG_ENDIAN,
        ),
)


FOR_FLOAT = dict(
    f16=(
        [1.244140625],
        b"\x3C\xFA",
        Code.F16,
        Code.BIG_ENDIAN,
    ),
    f32=(
        6547.525390625,
        b"\x45\xCC\x9C\x34",
        Code.F32,
        Code.BIG_ENDIAN,
    ),
    f64=(
        [3.141592653589793],
        b"\x40\x09\x21\xFB\x54\x44\x2D\x18",
        Code.F64,
        Code.BIG_ENDIAN,
    ),
)


class TestBytesIntTransformers(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            BytesIntEncoder(Code.U16),
            word_bytesize=2,
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid fmt"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                BytesIntEncoder(Code.F32)
            self.assertEqual(
                "'fmt' option <Code.F32: 529> not allowed",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid order"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                BytesIntEncoder(order=Code.U8)
            self.assertEqual(
                "'order' option <Code.U8: 520> not allowed",
                exc.exception.args[0],
            )

    def test_decode(self) -> None:
        for name, (decoded, encoded, *args) in FOR_INT.items():
            with self.subTest(test=name):
                assert_array_almost_equal(
                    decoded, BytesIntDecoder(*args).decode(encoded),
                )

    def test_encode(self) -> None:
        for name, (decoded, encoded, *args) in FOR_INT.items():
            with self.subTest(test=name):
                self.assertEqual(
                    encoded, BytesIntEncoder(*args).encode(decoded),
                )


class TestBytesFloatTransformers(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            BytesFloatEncoder(),
            word_bytesize=4,
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid fmt"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                BytesFloatEncoder(Code.U8)
            self.assertEqual(
                "'fmt' option <Code.U8: 520> not allowed",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid order"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                BytesFloatEncoder(order=Code.F32)
            self.assertEqual(
                "'order' option <Code.F32: 529> not allowed",
                exc.exception.args[0],
            )

    def test_decode(self) -> None:
        for name, (decoded, encoded, *args) in FOR_FLOAT.items():
            with self.subTest(test=name):
                assert_array_almost_equal(
                    decoded, BytesFloatDecoder(*args).decode(encoded),
                )

    def test_encode(self) -> None:
        for name, (decoded, encoded, *args) in FOR_FLOAT.items():
            with self.subTest(test=name):
                self.assertEqual(
                    encoded, BytesFloatEncoder(*args).encode(decoded),
                )


class TestGetBytesTransformers(unittest.TestCase):

    def test_basic_usage(self) -> None:
        for fmt, (ref_dec, ref_enc) in {
            Code.U8: (BytesIntDecoder, BytesIntEncoder),
            Code.F16: (BytesFloatDecoder, BytesFloatEncoder),
        }.items():
            with self.subTest(fmt=fmt):
                act_dec, act_enc = get_bytes_transformers(fmt)
                self.assertIsInstance(act_dec, ref_dec)
                self.assertIsInstance(act_enc, ref_enc)

    def test_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            get_bytes_transformers(Code.NONE)
        self.assertEqual(
            "unsupported format: <Code.NONE: 0>", exc.exception.args[0]
        )

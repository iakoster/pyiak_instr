import unittest

from src.pyiak_instr.testing import validate_object

from src.pyiak_instr.utilities import BasicByteStuffingCodec


class TestBasicByteStuffingCodec(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            BasicByteStuffingCodec(),
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="large stuff_byte"):
            with self.assertRaises(ValueError) as exc:
                BasicByteStuffingCodec(stuff_byte=b"\x00\x00")
            self.assertEqual(
                "'stuff_byte' must be a single byte",
                exc.exception.args[0],
            )

        with self.subTest(test="negative stuff_length"):
            with self.assertRaises(ValueError) as exc:
                BasicByteStuffingCodec(stuff_length=-1)
            self.assertEqual(
                "'stuff_length' must be at least 1",
                exc.exception.args[0],
            )

    def test_decode(self) -> None:
        act, act_dict = BasicByteStuffingCodec().decode(
            b"\x00\x00\x00\x01\x32\x00\x11\x33\x00\x00\x01\x00\x04\x33"
        )
        self.assertEqual(b"\x00\x32\x33\x00\x01\x33", act)
        self.assertDictEqual({1: b"\x01", 2: b"\x11", 5: b"\x04"}, act_dict)

        act, act_dict = BasicByteStuffingCodec(stuff_length=2).decode(
            b"\x00\x00\x00\x00\x01\x32\x11\x33\x00\x00\x00\x00\x10\x00\x33"
        )
        self.assertEqual(b"\x00\x11\x33\x00\x33", act)
        self.assertDictEqual({1: b"\x01\x32", 4: b"\x10\x00"}, act_dict)

    def test_decode_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            BasicByteStuffingCodec(stuff_length=2).decode(b"\x02\x01\x00\x01")
        self.assertEqual(
            "stuffing additional bytes do not have required length",
            exc.exception.args[0],
        )

    def test_encode(self) -> None:
        self.assertEqual(
            b"\x00\x00\x02\x00\x00\x33\x44\x53\x00\x00\x11",
            BasicByteStuffingCodec().encode(b"\x00\x02\x00\x33\x44\x53\x00\x11"),
        )

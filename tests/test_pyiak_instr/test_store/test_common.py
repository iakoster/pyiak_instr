import unittest

import numpy as np
from numpy.testing import assert_array_equal

from src.pyiak_instr.store import BitVector

from ...utils import validate_object


class TestBitVector(unittest.TestCase):

    def test_init(self) -> None:
        bv = self._get_bv()
        validate_object(
            self,
            bv,
            wo_attrs=["values"],
            bit_count=50,
        )
        assert_array_equal([0] * 7, bv.values)

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            BitVector(0)
        self.assertEqual(
            "The bits count cannot be less than 1", exc.exception.args[0]
        )

    def test_get_set(self) -> None:
        bv = self._get_bv(12)

        bv.set(4, 1)
        assert_array_equal([0x10, 0], bv.values)

        bv.set(6, True)
        assert_array_equal([0x50, 0], bv.values)

        bv[10] = 1
        assert_array_equal([0x50, 0x4], bv.values)

        bv[1] = True
        assert_array_equal([0x52, 0x4], bv.values)

        self.assertTrue(bv.get(4))

        bv.up(11)
        assert_array_equal([0x52, 0xC], bv.values)

        bv.down(11)
        assert_array_equal([0x52, 0x4], bv.values)

        bv[4] = False
        assert_array_equal([0x42, 0x4], bv.values)

        self.assertFalse(bv[4])

    def test_set_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            self._get_bv().set(10, 2)
        self.assertEqual(
            "bit value not in {0, 1, False, True}", exc.exception.args[0]
        )

    def test_get_coords(self) -> None:
        bv = self._get_bv(24)
        for index, ref in (
            (0, (0, 0)),
            (1, (0, 1)),
            (4, (0, 4)),
            (7, (0, 7)),
            (8, (1, 0)),
            (15, (1, 7)),
            (-1, (-1, 7)),
            (-4, (-1, 4)),
            (-8, (-1, 0)),
            (-16, (-2, 0)),
            (23, (2, 7)),
            (-24, (-3, 0)),
        ):
            with self.subTest(index=index, ref=ref):
                self.assertTupleEqual(ref, bv._get_coords(index))

    def test_get_coords_exc(self) -> None:
        bv = self._get_bv(12)
        for index in (12, -13):
            with self.subTest(index=index):
                with self.assertRaises(IndexError) as exc:
                    bv._get_coords(index)
                self.assertEqual(
                    "bit index out of range", exc.exception.args[0]
                )

    def test_values_setter(self):
        bv = self._get_bv(12)
        bv.set(2, 1)
        assert_array_equal([0x4, 0], bv.values)

        bv.values = [255, 255]
        assert_array_equal([0xFF, 0xF], bv.values)

    def test_values_setter_exc(self) -> None:
        bv = self._get_bv(12)
        with self.assertRaises(ValueError) as exc:
            bv.values = [0, 0, 0]
        self.assertEqual(
            "Invalid array length, 2 required", exc.exception.args[0]
        )

        with self.assertRaises(ValueError) as exc:
            bv.values = [0]
        self.assertEqual(
            "Invalid array length, 2 required", exc.exception.args[0]
        )

    @staticmethod
    def _get_bv(bits_count: int = 50):
        return BitVector(bits_count)

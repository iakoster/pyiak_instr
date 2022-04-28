import unittest

import numpy as np

from pyinstr_iakoster.store import BitVector


class TestBitVector(unittest.TestCase):

    @staticmethod
    def get_bv(bits_count: int = 10):
        return BitVector(bits_count)

    def compare_arrays(self, array_exp, array_act: np.ndarray):
        if len(array_exp) != len(array_act):
            raise AssertionError(
                f'Different length\nExpected: {len(array_exp)}\n'
                f'Actual: {len(array_act)}')
        for i, val in enumerate(array_exp):
            self.assertEqual(val, array_act[i])

    def setUp(self) -> None:
        self.bv = self.get_bv()

    def test_init_len(self):
        def compare_arrays(expected):
            if len(expected) != len(bv.values):
                raise AssertionError(
                    f'Different length\nExpected: {len(expected)}\n'
                    f'Actual: {len(bv.values)}')
            for i, val in enumerate(expected):
                self.assertEqual(val, bv.values[i])

        with self.assertRaises(ValueError) as exc:
            bv = self.get_bv(0)
        self.assertEqual(
            'The number of bits cannot be less than 1, got 0',
            exc.exception.args[0]
        )
        bv = self.get_bv(7)
        compare_arrays([0])
        self.assertEqual(7, bv.bit_count)
        bv = self.get_bv(8)
        compare_arrays([0])
        bv = self.get_bv(9)
        compare_arrays([0, 0])

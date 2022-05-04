import unittest

import numpy as np

from pyinstr_iakoster.store import BitVector


class TestBitVector(unittest.TestCase):

    @staticmethod
    def get_bv(bits_count: int = 50):
        return BitVector(bits_count)

    def compare_arrays(self, array_exp, array_act=None):
        if array_act is None:
            array_act = self.bit_vector.values
        if isinstance(array_exp, BitVector):
            array_exp = array_exp.values

        if len(array_act) != len(array_exp):
            raise AssertionError(
                f'Different length\nExpected: {len(array_act)}\n'
                f'Actual: {len(array_exp)}')
        for i, val in enumerate(array_act):
            self.assertEqual(array_exp[i], val, f'invalid {i} value')

    def setUp(self) -> None:
        self.bit_vector = self.get_bv()

    def test_init_len(self):
        def compare_arrays(expected):
            self.compare_arrays(bv, expected)

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

    def test_values_setter(self):
        self.compare_arrays([0] * 7)
        vals = ([1] * 7, [255] * 7)
        exp = ([1] * 7, [3] + [255] * 6)
        for i_test, (values, result) in enumerate(zip(vals, exp)):
            with self.subTest(test=i_test):
                self.bit_vector.values = values
                self.compare_arrays(result)

    def test_values_setter_wrong_shape(self):
        with self.assertRaises(ValueError) as exc:
            self.bit_vector.values = [1] * 6
        self.assertEqual(
            'Invalid shape of the new values: (6,) != (7,)',
            exc.exception.args[0])

    def test_bit_count_setter(self):
        self.bit_vector.values = [255] * 7

        bit_counts = (50, 24, 25, 23, 20, 24, 1, 1000)
        exp = ([3] + [255] * 6, [255] * 3, [0] + [255] * 3,
               [127] + [255] * 2, [15] + [255] * 2,
               [15] + [255] * 2, [1], [0] * 124 + [1])
        for i_test, (bit_count, result) in \
                enumerate(zip(bit_counts, exp)):
            with self.subTest(test=i_test):
                self.bit_vector.bit_count = bit_count
                self.compare_arrays(result)

    def test_bit_count_setter_wrong_count(self):
        with self.assertRaises(ValueError) as exc:
            self.bit_vector.bit_count = -1
        self.assertEqual(
            'The number of bits cannot be less than 1, got -1',
            exc.exception.args[0])
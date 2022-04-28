import unittest

import numpy as np

from pyinstr_iakoster.store import BitVector


class TestBitVector(unittest.TestCase):

    @staticmethod
    def get_bv(bits_count: int = 50):
        return BitVector(bits_count)

    def compare_arrays(self, array_act, array_exp=None):
        if array_exp is None:
            array_exp = self.bit_vector.values
        if isinstance(array_act, BitVector):
            array_act = array_act.values

        if len(array_exp) != len(array_act):
            raise AssertionError(
                f'Different length\nExpected: {len(array_exp)}\n'
                f'Actual: {len(array_act)}')
        for i, val in enumerate(array_exp):
            self.assertEqual(val, array_act[i])

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
        self.bit_vector.values = [1] * 7
        self.compare_arrays([1] * 7)

    def test_values_setter_wrong_shape(self):
        with self.assertRaises(ValueError) as exc:
            self.bit_vector.values = [1] * 6
        self.assertEqual(
            'Invalid shape of the new values: (6,) != (7,)',
            exc.exception.args[0])

    def test_bit_count_setter(self):
        with self.assertRaises(ValueError) as exc:
            self.bit_vector.bit_count = -1
        self.assertEqual(
            'The number of bits cannot be less than 1, got -1',
            exc.exception.args[0])

        self.bit_vector.values = list(range(7))
        self.assertTrue(
            (self.bit_vector.values == [0, 1, 2, 3, 4, 5, 6]).all())

        self.bit_vector.bit_count = 16
        self.assertTrue((self.bit_vector.values == [5, 6]).all())
        self.bit_vector.bit_count = 9
        print(self.bit_vector.values)
        self.bit_vector.bit_count = 16
        self.assertTrue((self.bit_vector.values == [5, 6]).all())
        self.bit_vector.bit_count = 17
        self.assertTrue((self.bit_vector.values == [0, 5, 6]).all())
        #self.assertTrue((self.bit_vector.values == [0, 5, 6]).all())

import unittest

import numpy as np

from pyiak_instr.store import BitVector


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

    def test_get_bit_small(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [103, 220]
        bit_array = np.unpackbits(
            np.array([103, 220], dtype=np.uint8), bitorder='big')[::-1]
        for i_test, index in enumerate(range(16)):
            with self.subTest(test=i_test, index=index):
                self.assertEqual(
                    bit_array[index],
                    self.bit_vector.get_bit(index))

    def test_get_bit(self):
        self.bit_vector.bit_count = 64
        values = np.uint8(np.random.uniform(0, 255, size=8))
        self.bit_vector.values = values
        bit_array = np.unpackbits(values)[::-1]
        bit_indexes = np.uint8(np.round(np.random.uniform(0, 63, size=50)))
        for i_test, index in enumerate(bit_indexes):
            with self.subTest(test=i_test, index=index):
                self.assertEqual(
                    bit_array[index],
                    self.bit_vector.get_bit(index))

    def test_get_bit_index_error(self):
        with self.assertRaises(IndexError):
            self.bit_vector.get_bit(50)
        with self.assertRaises(IndexError) as exc:
            self.bit_vector.get_bit(-51)
        self.bit_vector.get_bit(-50)
        self.assertEqual(
            'bit index out of range',
            exc.exception.args[0])

    def test_get_flag_small(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [103, 220]
        bit_array = np.bool_(np.unpackbits(
            np.array([103, 220], dtype=np.uint8), bitorder='big'))[::-1]
        for i_test, index in enumerate(range(16)):
            with self.subTest(test=i_test, index=index):
                result = bit_array[index] ^ \
                         self.bit_vector.get_flag(index)
                self.assertIsInstance(result, (bool, np.bool_))
                self.assertFalse(result)

    def test_set_bit_small(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [0, 0]
        bit_array = np.unpackbits(
            np.array([103, 220], dtype=np.uint8), bitorder='big')[::-1]
        for index in range(16):
            self.bit_vector.set_bit(index, bit_array[index])
        self.compare_arrays([103, 220])

    def test_set_bit_invalid_value(self):
        with self.assertRaises(ValueError) as exc:
            self.bit_vector.set_bit(0, 2)
        self.assertEqual(
            'invalid bit value, expected only one '
            'of {0, 1, False, True}',
            exc.exception.args[0])

    def test_set_flag_small(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [0, 0]
        bit_array = np.bool_(np.unpackbits(
            np.array([103, 220], dtype=np.uint8), bitorder='big'))[::-1]
        for index in range(16):
            self.bit_vector.set_bit(index, bit_array[index])
        self.compare_arrays([103, 220])

    def test_raise_flag(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [0, 0]
        self.bit_vector.raise_flag(5)
        self.assertEqual(32, self.bit_vector.values[-1])

    def test_lower_flag(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [255, 255]
        self.bit_vector.lower_flag(5)
        self.assertEqual(223, self.bit_vector.values[-1])

    def test_getitem(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [230, 72]
        bit_array = np.bool_(np.unpackbits(
            np.array([230, 72], dtype=np.uint8), bitorder='big'))[::-1]
        for i_test, index in enumerate(range(16)):
            with self.subTest(test=i_test, index=index):
                result = bit_array[index] ^ self.bit_vector[index]
                self.assertIsInstance(result, (bool, np.bool_))
                self.assertFalse(result)
        self.compare_arrays([230, 72])

    def test_setitem(self):
        self.bit_vector.bit_count = 16
        self.bit_vector.values = [0, 0]
        bit_array = np.bool_(np.unpackbits(
            np.array([230, 72], dtype=np.uint8), bitorder='big'))[::-1]
        for index in range(16):
            self.bit_vector[index] = bit_array[index]
        self.compare_arrays([230, 72])

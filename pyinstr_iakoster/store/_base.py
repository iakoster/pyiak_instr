import numpy as np


__all__ = ['BitVector']


class BitVector(object):
    """
    Class for bitwise operations on an array as
    a large single value.

    The vector contains values in the numpy array, where
    it stores each bit in unsigned 8-bit integer values.

    Parameters
    ----------
    bit_count: int
        length of the vector in bits.

    Raises
    ------
    ValueError
        if bit_count is less than 1.
    """

    VALUE_CAPACITY = 8

    def __init__(self, bit_count: int):
        if bit_count < 1:
            raise ValueError(
                'The number of bits cannot be less than 1, '
                'got %d' % bit_count)

        vals_count = bit_count // self.VALUE_CAPACITY
        if bit_count % self.VALUE_CAPACITY:
            vals_count += 1
        self._bit_c = bit_count
        self._vals = np.zeros(vals_count, dtype=np.uint8)

    @property
    def values(self) -> np.ndarray:
        """Array of the values in the vector"""
        return self._vals

    @property
    def bit_count(self) -> int:
        """length of the vector in bits"""
        return self._bit_c

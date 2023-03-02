"""Private module of ``pyiak_instr.store`` with common store classes"""
import numpy as np
import numpy.typing as npt


__all__ = ["BitVector"]


class BitVector:
    """
    Class for bitwise operations on an array as a large single value.

    The vector contains values in the numpy array, where it stores each bit
    in unsigned 8-bit integer values.

    Parameters
    ----------
    bits_count: int
        length of the vector in bits (count of valid bits in the vector).

    Raises
    ------
    ValueError
        if `bit_count` is less than 1.
    """

    BITS_IN_VALUE = 8

    def __init__(self, bits_count: int):
        if bits_count < 1:
            raise ValueError("The bits count cannot be less than 1")

        vals, mod = divmod(bits_count, self.BITS_IN_VALUE)
        if mod:
            vals += 1
        self._bits = bits_count
        self._vals: npt.NDArray[np.uint8] = np.zeros(vals, dtype=np.uint8)

    def down(self, index: int) -> None:
        """
        Set False flag to bit by index.

        Parameters
        ----------
        index : int
            bit index.
        """
        self.set(index, False)

    def get(self, index: int) -> bool:
        """
        Get the bit flag by index.

        Parameters
        ----------
        index : int
            bit index.

        Returns
        -------
        bool
            flag.
        """
        return bool(self.get_bit(index))

    def get_bit(self, index: int) -> int:
        """
        Get the bit value by index.

        Parameters
        ----------
        index : int
            bit index.

        Returns
        -------
        int
            bit value.
        """
        i_val, i_bit = self._get_coords(index)
        return self._vals[i_val] >> i_bit & 1  # type: ignore

    def set(self, index: int, bit: int | bool) -> None:
        """
        Set new bit value by index.

        Parameters
        ----------
        index : int
            bit index.
        bit : int | bool
            bit value.

        Raises
        ------
        ValueError
            if bit not in {0, 1, False, True}
        """
        if bit not in (0, 1, False, True):
            raise ValueError("bit value not in {0, 1, False, True}")
        i_val, i_bit = self._get_coords(index)
        if bit:
            self._vals[i_val] |= 1 << i_bit
        else:
            self._vals[i_val] &= ~(1 << i_bit)
        # universal: (self._vals[i_val] & ~(1 << i_bit)) | (bit << i_bit)

    # pylint: disable=invalid-name
    def up(self, index: int) -> None:
        """
        Set True flag to bit by index.

        Parameters
        ----------
        index : int
            bit index.
        """
        self.set(index, True)

    def _get_coords(self, index: int) -> tuple[int, int]:
        """
        Get indexes for value and bit.

        Parameters
        ----------
        index : int
            bit index.

        Returns
        -------
        i_val: int
            value index.
        i_bit: int
            bit index in value.

        Raises
        ------
        IndexError
            if `index` more that `bits_count`.
        """
        if not -self._bits <= index < self._bits:
            raise IndexError("bit index out of range")
        return divmod(index, self.BITS_IN_VALUE)

    @property
    def values(self) -> npt.NDArray[np.uint8]:
        """
        Returns
        -------
        NDArray
            Copy of the array with values in the vector.
        """
        return self._vals.copy()

    @values.setter
    def values(
        self,
        values: npt.NDArray[np.uint8] | list[int] | tuple[int, ...] | int,
    ) -> None:
        """
        Set new values to the vector.

        Parameters
        ----------
        values : Collection
            new values for vector.

        Raises
        ------
        ValueError
            if new values have different length with current array.
        """
        if not isinstance(values, int) and len(values) != self._vals.shape[0]:
            raise ValueError(
                "Invalid array length, %d required" % self._vals.shape[0]
            )

        new = np.array(values, dtype=np.uint8)
        bits_mod = self._bits % self.BITS_IN_VALUE
        if bits_mod:
            new[-1] &= 0xFF >> self.BITS_IN_VALUE - bits_mod
        self._vals = new

    @property
    def bit_count(self) -> int:
        """
        Returns
        -------
        int
            bits count in vector.
        """
        return self._bits

    def __getitem__(self, index: int) -> bool:
        """
        Get flag from index.

        Parameters
        ----------
        index : int
            bit index.

        Returns
        -------
        bool
            bit flag.
        """
        return self.get(index)

    def __setitem__(self, index: int, flag: int | bool) -> None:
        """
        Set value by index.

        Parameters
        ----------
        index : int
            bit index.
        flag : int | bool
            new value for bit.
        """
        self.set(index, flag)

import unittest

import random


from src.pyiak_instr.utilities import num_sign, to_base


class TestNumSign(unittest.TestCase):

    def test_num_sign(self) -> None:
        data = (
            (1, 1, False),
            (-1, -1, False),
            (0, 0, False),
            (0, 1, True),
            (100, 1, False),
            (-100, -1, False),
        )
        for i_test, (value, ref, pos_zero) in enumerate(data):
            with self.subTest(test=i_test):
                self.assertEqual(ref, num_sign(value, pos_zero=pos_zero))


class TestToBase(unittest.TestCase):

    def test_to_base(self) -> None:
        to_base(20, 16)

    def test_to_base_random(self) -> None:
        for i in range(50):
            value, base = random.randint(-10**9, 10**9), random.randint(2, 36)
            with self.subTest(test=i, value=value, base=base):
                self.assertEqual(value, int(to_base(value, base), base))

    def test_to_base_exc(self) -> None:
        bases = (1, 37)
        for base in bases:
            with self.subTest(base=base):
                with self.assertRaises(ValueError) as exc:
                    to_base(10, base)
                self.assertEqual(
                    "base must be in range [2; 36]", exc.exception.args[0]
                )

import unittest

import random

from pyinstr_iakoster.utilities import to_base


class TestBaseOperations(unittest.TestCase):

    chr_ids = list(range(48, 58)) + list(range(97, 123))

    def test_to_base(self):
        self.assertListEqual(
            [to_base(i, 36) for i in range(0, 36)],
            [chr(i) for i in self.chr_ids]
        )

    def test_to_base_negatives(self):
        self.assertListEqual(
            [to_base(i, 36) for i in range(-35, 0)],
            ["-" + chr(i) for i in self.chr_ids[1:][::-1]]
        )

    def test_to_base_large(self):
        values = [int(random.uniform(36, 10000)) for _ in range(100)]
        self.assertListEqual(
            [to_base(v, 16) for v in values],
            [hex(v)[2:] for v in values]
        )

    def test_to_base_exceptions(self):
        with self.assertRaises(ValueError) as exc:
            to_base(0, 37)
        self.assertEqual(
            exc.exception.args[0],
            "base must be in range [2; 36]"
        )

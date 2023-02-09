import unittest

import random

from pyiak_instr.utilities import to_base, from_base


class TestBaseOperations(unittest.TestCase):

    chr_ids = [chr(c) for c in list(range(48, 58)) + list(range(97, 123))]

    def test_to_base(self):
        self.assertListEqual(
            [to_base(i, 36) for i in range(0, 36)],
            self.chr_ids
        )

    def test_to_base_negatives(self):
        self.assertListEqual(
            [to_base(i, 36) for i in range(-35, 0)],
            ["-" + i for i in self.chr_ids[1:][::-1]]
        )

    def test_to_base_large(self):
        values = [int(random.uniform(0, 10000)) for _ in range(100)]
        self.assertListEqual(
            [to_base(v, 16) for v in values],
            [hex(v)[2:] for v in values]
        )
        values = [int(random.uniform(-10000, -10)) for _ in range(100)]
        self.assertListEqual(
            [to_base(v, 16) for v in values],
            ["-" + hex(v)[3:] for v in values]
        )

    def test_to_base_exceptions(self):
        with self.assertRaises(ValueError):
            to_base(0, 37)
        with self.assertRaises(ValueError) as exc:
            to_base(0, 1)
        self.assertEqual(
            exc.exception.args[0],
            "base must be in range [2; 36]"
        )

    def test_from_base(self):
        self.assertListEqual(
            [from_base(i, 36) for i in self.chr_ids],
            list(range(36))
        )

    def test_from_base_negatives(self):
        self.assertListEqual(
            [from_base("-" + i, 36) for i in self.chr_ids],
            list(range(0, -36, -1))
        )

    def test_from_base_large(self):
        values = [int(random.uniform(-10000, 10000)) for _ in range(100)]
        self.assertListEqual(
            [from_base(to_base(v, 36), 36) for v in values],
            values
        )

    def test_from_base_exceptions(self):
        with self.assertRaises(ValueError):
            from_base("0", 37)
        with self.assertRaises(ValueError) as exc:
            from_base("0", 1)
        self.assertEqual(
            exc.exception.args[0],
            "base must be in range [2; 36]"
        )

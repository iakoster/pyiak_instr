import unittest

from src.pyiak_instr.exceptions import ContentError


class TestContentError(unittest.TestCase):

    def test_args(self) -> None:
        self.assertTupleEqual(
            ("invalid content in int: test",),
            ContentError(int(), clarification="test").args,
        )
        self.assertEqual(
            "invalid content in int", ContentError(int()).msg
        )

import unittest

from src.pyiak_instr.typing import WithBaseStringMethods


class WithBaseStringMethodsTestInstance(WithBaseStringMethods):

    def __str_under_brackets__(self) -> str:
        return "test=5"


class TestWithBaseStringMethods(unittest.TestCase):

    def test_magic_repr(self) -> None:
        self.assertEqual(
            "<WithBaseStringMethodsTestInstance(test=5)>",
            repr(WithBaseStringMethodsTestInstance())
        )

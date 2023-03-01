import unittest

from src.pyiak_instr.exceptions import PyiError


class TestPyiError(unittest.TestCase):

    def test_init(self) -> None:
        with self.assertRaises(PyiError) as exc:
            raise PyiError(1, [23], msg="test case")
        res = exc.exception
        self.assertTupleEqual(("test case", 1, [23]), res.args)
        self.assertEqual("test case", res.msg)
        self.assertEqual("test case", str(res))
        self.assertEqual("PyiError: test case", repr(res))

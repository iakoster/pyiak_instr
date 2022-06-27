import unittest
import traceback
import pandas.errors

from pyinstr_iakoster.exceptions import (
    PyiError
)


class TestPyiError(unittest.TestCase):

    def setUp(self) -> None:
        self.exc = PyiError("test msg", "test_arg", 1, 2.3)

    def test_init(self):
        self.assertTupleEqual(
            ("test msg", "test_arg", 1, 2.3),
            self.exc.args
        )
        self.assertEqual("test msg", self.exc.message)

    def test_magic_repr(self):
        self.assertEqual("PyiError: test msg", repr(self.exc))

    def test_magic_str(self):
        self.assertEqual("test msg", str(self.exc))

    def test_traceback(self):
        try:
            raise self.exc
        except PyiError as exc:
            self.assertEqual(
                "pyinstr_iakoster.exceptions._base.PyiError: test msg",
                traceback.format_exception(exc)[-1][:-1]
            )

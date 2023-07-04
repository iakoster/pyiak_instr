import unittest
from inspect import currentframe

from src.pyiak_instr.core import Code

from src.pyiak_instr.exceptions import (
    CodeNotAllowed,
    NotConfiguredYet,
    NotAmongTheOptions,
    NotSupportedMethod,
    WithoutParent,
)


class TestCodeNotAllowed(unittest.TestCase):

    def test_args(self) -> None:
        self.assertTupleEqual(
            ("'code' option <Code.CODE: 258> not allowed",),
            CodeNotAllowed(Code.CODE).args,
        )


class TestNotConfiguredYet(unittest.TestCase):

    def test_args(self) -> None:
        self.assertTupleEqual(
            ("int not configured yet",), NotConfiguredYet(3).args
        )


class TestNotAmongTheOptions(unittest.TestCase):

    def test_args(self) -> None:
        ref = dict(
            case_1=("'test' option not allowed",),
            case_2=("'test' option 3 not in {1, 2}",),
        )
        init = dict(
            case_1=dict(
                name="test"
            ),
            case_2=dict(
                name="test",
                value=3,
                options={1, 2}
            )
        )

        sym_diff = set(ref).symmetric_difference(set(init))
        assert len(sym_diff) == 0, f"case difference: {sym_diff}"
        for (test, msg), (_, init_kw) in zip(ref.items(), init.items()):
            self.assertTupleEqual(msg, NotAmongTheOptions(**init_kw).args)


class TestNotSupportedFunction(unittest.TestCase):

    def test_init(self) -> None:
        cases = [
            ("not supported method", None),
            (
                "TestNotSupportedFunction does not support .test_init",
                currentframe(),
            ),
            ("AnyClass does not support .any_method", "AnyClass.any_method"),
        ]

        for i_case, (msg, frame) in enumerate(cases):
            with self.subTest(case=i_case):
                self.assertEqual(msg, NotSupportedMethod(frame).msg)

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid qualname"):
            with self.assertRaises(ValueError) as exc:
                NotSupportedMethod("Class.invalid.qualname")
            self.assertEqual(
                "invalid method qualname: 'Class.invalid.qualname'",
                exc.exception.args[0],
            )

        with self.subTest(test="invalid frame type"):
            with self.assertRaises(TypeError) as exc:
                NotSupportedMethod(1)
            self.assertEqual(
                "invalid frame type: <class 'int'>", exc.exception.args[0]
            )


class TestWithoutParent(unittest.TestCase):

    def test_args(self) -> None:
        self.assertTupleEqual(("parent not specified",), WithoutParent().args)


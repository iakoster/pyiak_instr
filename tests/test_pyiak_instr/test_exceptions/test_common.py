import unittest

from src.pyiak_instr.core import Code

from src.pyiak_instr.exceptions import (
    CodeNotAllowed,
    NotConfiguredYet,
    NotAmongTheOptions,
    WithoutParent,
)


class TestCodeNoaAllowed(unittest.TestCase):

    def test_args(self) -> None:
        self.assertTupleEqual(
            ("code option not allowed, got <Code.CODE: 258>",),
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
            case_1=("test option not allowed",),
            case_2=("test option not in {1, 2}, got 3",),
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


class TestWithoutParent(unittest.TestCase):

    def test_args(self) -> None:
        self.assertTupleEqual(("parent not specified",), WithoutParent().args)


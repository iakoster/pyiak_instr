import unittest

from src.pyiak_instr.testing import get_members, get_object_attrs


class TestInstance:

    DECLARED_CONSTANT: int
    declared_variable: int
    CONSTANT = 0
    class_variable = 0

    def __init__(self):
        self.variable = 0

    def callable_error(self) -> None:
        raise AttributeError()

    def callable(self) -> None:
        return

    def _protected(self) -> None:
        return

    def __private(self) -> None:
        return

    @property
    def prop(self) -> int:
        return 0

    @property
    def prop_error(self) -> None:
        raise AttributeError()

    def __magic__(self) -> None:
        return


class TestGetMembers(unittest.TestCase):

    members = {
                "__annotations__",
                "__dir__",
                "__le__",
                "__weakref__",
                "__new__",
                "__str__",
                "__setattr__",
                "__init_subclass__",
                "__module__",
                "__subclasshook__",
                "__getstate__",
                "__getattribute__",
                "__init__",
                "__eq__",
                "__hash__",
                "__format__",
                "__repr__",
                "__doc__",
                "__dict__",
                "__ne__",
                "__reduce__",
                "__class__",
                "__gt__",
                "__reduce_ex__",
                "__ge__",
                "__sizeof__",
                "__delattr__",
                "__lt__",
                "CONSTANT",
                "class_variable",
                "variable",
                "callable",
                "callable_error",
                "_protected",
                "_TestInstance__private",
                "prop",
                "__magic__",
            }

    def test_basic(self) -> None:
        members = {n for n, _ in get_members(TestInstance(), [])}
        self.assertSetEqual(self.members, members)

    def test_with_pass(self) -> None:
        members = {n for n, _ in get_members(TestInstance(), ["prop", "not_existed"])}
        self.assertSetEqual(self.members - {"prop"}, members)


class TestGetObjectAttrs(unittest.TestCase):

    def test_basic(self) -> None:
        self.assertListEqual(
            ["class_variable", "prop", "variable"],
            get_object_attrs(TestInstance()),
        )

    def test_without_some(self) -> None:
        self.assertListEqual(
            ["class_variable", "variable"],
            get_object_attrs(TestInstance(), wo_attrs=["prop", "not_existed"]),
        )

    def test_with_constants(self) -> None:
        self.assertListEqual(
            ["CONSTANT", "class_variable", "prop", "variable"],
            get_object_attrs(TestInstance(), wo_consts=False),
        )

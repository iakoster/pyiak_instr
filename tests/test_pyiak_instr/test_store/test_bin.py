import unittest

import numpy as np

from ...utils import validate_object, compare_values

from src.pyiak_instr.store import BytesField, BytesFieldPattern


class TestBytesField(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            BytesField(
                start=4,
                may_be_empty=True,
                fmt="I",
                order=">",
                expected=-100,
            ),
            bytes_expected=0,
            default=b"",
            expected=0,
            fmt="I",
            infinite=True,
            may_be_empty=True,
            order=">",
            slice=slice(4, None),
            start=4,
            stop=None,
            word_size=4,
            check_attrs=True,
        )

    def test_decode(self) -> None:
        obj = BytesField(
            start=4,
            may_be_empty=True,
            fmt="I",
            order=">",
            expected=-100,
        )
        cases = (
            (b"\x00\x00\x00\x01", 1),
            (b"\x00\x00\x00\x01\x00\x00\x00\x02", [1, 2]),
            (b"\x00\x00\x00\x01\x00\x00\x00\x22", np.array([1, 0x22])),
        )
        for i_case, (data, ref) in enumerate(cases):
            with self.subTest(test=i_case):
                compare_values(self, ref, obj.decode(data))

    def test_encode(self) -> None:
        obj = BytesField(
            start=4,
            may_be_empty=True,
            fmt="I",
            order=">",
            expected=-100,
        )
        cases = (
            (1, b"\x00\x00\x00\x01"),
            ([1, 2], b"\x00\x00\x00\x01\x00\x00\x00\x02"),
            (np.array([1, 0x22]), b"\x00\x00\x00\x01\x00\x00\x00\x22"),
        )
        for i_case, (data, ref) in enumerate(cases):
            with self.subTest(test=i_case):
                compare_values(self, ref, obj.encode(data))


class TestBytesFieldPattern(unittest.TestCase):

    def test_add(self) -> None:
        pattern = self._get_pattern()
        self.assertNotIn("e", pattern)
        pattern.add("e", 223)
        self.assertIn("e", pattern)

    def test_add_exc(self) -> None:
        with self.assertRaises(KeyError) as exc:
            self._get_pattern().add("a", 1)
        self.assertEqual("parameter in pattern already", exc.exception.args[0])

    def test_get(self) -> None:
        pattern = BytesFieldPattern(
            may_be_empty=True,
            fmt="B",
            order="",
            expected=4,
        )
        validate_object(
            self,
            pattern.get(start=4),
            start=4,
            may_be_empty=True,
            fmt="B",
            order="",
            expected=4,
            check_attrs=False,
        )

    def test_get_updated(self) -> None:
        pattern = BytesFieldPattern(
            may_be_empty=True,
            fmt="B",
            order="",
            expected=4,
        )
        with self.assertRaises(TypeError) as exc:
            pattern.get(start=0, expected=1)
        self.assertIn(
            "got multiple values for keyword argument 'expected'",
            exc.exception.args[0],
        )

        validate_object(
            self,
            pattern.get_updated(start=0, expected=1),
            start=0,
            expected=1,
            check_attrs=False,
        )

    def test_pop(self) -> None:
        pattern = self._get_pattern()
        self.assertIn("a", pattern)
        self.assertEqual(1, pattern.pop("a"))
        self.assertNotIn("a", pattern)

    def test_magic_contains(self) -> None:
        self.assertIn("a", self._get_pattern())

    def test_magic_getitem(self) -> None:
        self.assertEqual(1, self._get_pattern()["a"])

    def test_magic_setitem(self) -> None:
        pattern = self._get_pattern()
        self.assertListEqual([], pattern["b"])
        pattern["b"] = 1
        self.assertEqual(1, pattern["b"])

    def test_magic_setitem_exc(self) -> None:
        pattern = self._get_pattern()
        with self.assertRaises(KeyError) as exc:
            pattern["e"] = 1
        self.assertEqual("'e' not in parameters", exc.exception.args[0])

    @staticmethod
    def _get_pattern() -> BytesFieldPattern:
        return BytesFieldPattern(
            a=1,
            b=[],
            c={},
            d="string"
        )

import unittest

from tests.env_vars import TEST_DATA_DIR
from ..utils import (
    get_mf_n0,
    get_mf_n1,
    validate_object,
)

from pyinstr_iakoster.communication import (
    AsymmetricResponseField,
)


DATA_JSON_PATH = TEST_DATA_DIR / "test.json"
DATA_DB_PATH = TEST_DATA_DIR / "test.db"


class TestAsymmetricResponseField(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.arf = {
            "empty": AsymmetricResponseField(),
            0: AsymmetricResponseField(
                operand="==",
                value=b"\x4c",
                start=0,
                stop=1,
            ),
            1: AsymmetricResponseField(
                operand="!=",
                value=b"\x4c\x54",
                start=1,
                stop=3,
            ),
            2: AsymmetricResponseField(
                operand="==",
                value="\\bts\t241,32",
                start=2,
                stop=4,
            )
        }

    def test_init(self) -> None:

        kwargs = dict(
            operand=("", "==", "!=", "=="),
            start=(None, 0, 1, 2),
            stop=(None, 1, 3, 4),
            value=(b"", b"\x4c", b"\x4c\x54", b"\xf1\x20"),
            is_empty=(True, False, False, False),
            kwargs=(
                {},
                dict(operand="==", start=0, stop=1, value=b"\x4c"),
                dict(operand="!=", start=1, stop=3, value=b"\x4c\x54"),
                dict(operand="==", start=2, stop=4, value=b"\xf1\x20")
            )
        )

        for i, (id_, field) in enumerate(self.arf.items()):
            kw = {k: v[i] for k, v in kwargs.items()}
            with self.subTest(id=id_):
                validate_object(
                    self, self.arf[id_], check_attrs=True, **kw
                )

    def test_match(self):

        messages = (
            b"\x4c\x4c\x54\x01\xfe",
            b"\xef\x4c\x54\x01\xed",
            b"\x12\x4c\xf1\x20\x9a",
        )
        res_message = {
            "empty": messages,
            0: [m[1:] for m in messages],
            1: [m[:1] + m[3:] for m in messages],
            2: [m[:2] + m[4:] for m in messages],
        }
        res_match = {
            "empty": (False, False, False),
            0: (True, False, False),
            1: (False, False, True),
            2: (False, False, True),
        }
        for key, arf in self.arf.items():
            for i_msg, msg in enumerate(messages):
                res_msg, res = arf.match(msg)

                with self.subTest(key=key, i_msg=i_msg, res="match"):
                    self.assertEqual(res_match[key][i_msg], res)
                with self.subTest(key=key, i_msg=i_msg, res="message"):
                    self.assertEqual(res_message[key][i_msg], res_msg)

    def test_init_exc(self) -> None:

        test_data: list[tuple[str, dict[str, str | int], Exception]] = [
            (
                "invalid operand",
                dict(operand="t"),
                ValueError("invalid operand: 't'"),
            ),
            (
                "without stop",
                dict(operand="==", start=10),
                ValueError("start or stop is not specified"),
            ),
            (
                "invalid range",
                dict(operand="==", start=10, stop=10),
                ValueError("stop <= start"),
            ),
            (
                "invalid type",
                dict(operand="==", start=0, stop=1, value=1),
                TypeError("invalid type of value"),
            ),
            (
                "invalid string",
                dict(operand="==", start=0, stop=1, value="test"),
                TypeError("value can't be converted from string to bytes"),
            )
        ]

        for name, kwargs, err in test_data:
            with self.subTest(test=name):
                with self.assertRaises(err.__class__) as exc:
                    AsymmetricResponseField(**kwargs)
                self.assertEqual(err.args[0], exc.exception.args[0])

    def test_match_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            self.arf[0].match("")
        self.assertEqual("bytes message required", exc.exception.args[0])


class TestMessageFormat(unittest.TestCase):

    def test_init(self):

        for mf, ref_data in (get_mf_n0(), get_mf_n1()):
            format_name = mf.msg_args["mf_name"]

            with self.subTest(format_name=format_name):
                self.assertDictEqual(ref_data["msg_args"], mf.msg_args)

            with self.subTest(format_name=format_name, setter="all"):
                self.assertEqual(len(ref_data["setters"]), len(mf.setters))
                for (ref_name, ref_setter), (name, setter) in zip(
                    ref_data["setters"].items(), mf.setters.items()
                ):
                    with self.subTest(format_name=format_name, setter=name):
                        self.assertEqual(ref_name, name)
                        self.assertEqual(
                            ref_setter["special"], setter.special
                        )
                        self.assertDictEqual(
                            ref_setter["kwargs"], setter.kwargs
                        )
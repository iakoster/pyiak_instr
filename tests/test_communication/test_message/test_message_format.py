import shutil
import unittest
from copy import deepcopy

from tests.env_vars import TEST_DATA_DIR
from ..data import (
    SETTERS,
    MF_DICT,
    MF_CFG_DICT,
    get_message,
    get_mf,
)
from ..utils import (
    validate_object,
    compare_objects,
    compare_messages
)

from pyinstr_iakoster.rwfile import RWConfig
from pyinstr_iakoster.communication import (
    AsymmetricResponseField,
    MessageFormat,
    MessageFormatMap,
)

TEST_DIR = TEST_DATA_DIR / __name__.split(".")[-1]
CFG_PATH = TEST_DIR / "cfg.ini"


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

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        with RWConfig(CFG_PATH) as rwc:
            rwc.write(MF_CFG_DICT)

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_init(self) -> None:

        for i_mf in range(2):
            mf, ref = get_mf(i_mf)
            mf_name = mf.message["mf_name"]

            with self.subTest(mf_name=mf_name, test="msg_args"):
                self.assertDictEqual(ref["message"], mf.message)

            with self.subTest(mf_name=mf_name, setter="all"):
                self.assertEqual(len(ref["setters"]), len(mf.setters))

                for (ref_name, ref_setter), (name, setter) in zip(
                    ref["setters"].items(), mf.setters.items()
                ):
                    with self.subTest(mf_name=mf_name, setter=name):
                        self.assertEqual(ref_name, name)
                        validate_object(self, setter, **ref_setter)
                    break

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            MessageFormat()
        self.assertIn(
            "missing the required setters: {'", exc.exception.args[0]
        )

    def test_read(self) -> None:
        for mf_name, ref_mf in MF_DICT.items():
            with self.subTest(mf_name=mf_name):
                mf = MessageFormat.read(CFG_PATH, mf_name)
                compare_objects(self, ref_mf, mf)

    def test_write(self) -> None:

        cfg_path = TEST_DIR / "cfg_test.ini"
        with RWConfig(cfg_path) as rwc:
            if "master" not in rwc.hapi.sections():
                rwc.hapi.add_section("master")
            rwc.set("master", "formats", "\\lst\tn0,n1,n2,n3")
            rwc.apply_changes()

        for ref_mf in MF_DICT.values():
            ref_mf.write(cfg_path)

        with open(CFG_PATH, "r") as ref_io, open(cfg_path, "r") as res_io:
            ref_lines, res_lines = ref_io.readlines(), res_io.readlines()
            self.assertEqual(len(ref_lines), len(res_lines))

            for line, (ref, res) in enumerate(zip(ref_lines, res_lines)):
                if line in (16, 31):  # fixme: i don't know why in this line error
                    self.assertEqual("arf = \\dct\n", res)
                    continue

                with self.subTest(line=line):
                    self.assertEqual(ref, res)

        for mf_name, ref in MF_DICT.items():
            with self.subTest(mf_name=mf_name):
                res = MessageFormat.read(cfg_path, mf_name)
                compare_objects(self, ref, res)

    def test_get(self) -> None:
        test_data = [
            (get_message(0), get_mf(0, get_ref=False).get()),
            (get_message(1), get_mf(1, get_ref=False).get()),
            (get_message(2), get_mf(2, get_ref=False).get()),
        ]
        for i_test, (ref, res) in enumerate(test_data):
            with self.subTest(i_test=i_test):
                compare_messages(self, ref, res)

    def test_get_with_update(self) -> None:
        setters = deepcopy(SETTERS[1])
        setters["data"].kwargs["fmt"] = ">I"
        compare_messages(
            self,
            get_message(1).configure(**setters),
            get_mf(1, get_ref=False).get(data={"fmt": ">I"})
        )

        setters = deepcopy(SETTERS[2])
        setters["data"].kwargs["fmt"] = "B"
        compare_messages(
            self,
            get_message(2).configure(**setters),
            get_mf(2, get_ref=False).get(data={"fmt": "B"})
        )

    def test_read_exc(self):
        with self.assertRaises(ValueError) as exc:
            MessageFormat.read(CFG_PATH, "n00")
        self.assertEqual(
            "format with name 'n00' not exists", exc.exception.args[0]
        )


class TestMessageFormatsMap(unittest.TestCase):

    maxDiff = None

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def setUp(self) -> None:
        with RWConfig(CFG_PATH) as rwc:
            rwc.write(MF_CFG_DICT)
        self.mf_map = MessageFormatMap(*MF_DICT.values())

    def test_init(self) -> None:
        pass  # There's nothing to test here yet

    def test_get(self) -> None:
        for mf_name, ref in MF_DICT.items():
            with self.subTest(mf_name=mf_name):
                compare_objects(self, ref, self.mf_map.get(mf_name))

    def test_read_write(self) -> None:
        cfg_path = TEST_DIR / "cfg_test.ini"

        mf_map = MessageFormatMap.read(CFG_PATH)
        for mf_name, mf in MF_DICT.items():
            with self.subTest(mf_name=mf_name):
                compare_objects(self, mf, mf_map.get(mf_name))

        mf_map.write(cfg_path)
        with open(CFG_PATH, "r") as ref_io, open(cfg_path, "r") as res_io:
            ref_lines, res_lines = ref_io.readlines(), res_io.readlines()
            self.assertEqual(len(ref_lines), len(res_lines))

            for line, (ref, res) in enumerate(zip(ref_lines, res_lines)):
                with self.subTest(line=line):

                    if line in (16, 31):  # fixme: i don't know why in this line error
                        self.assertEqual("arf = \\dct\n", res)
                        continue
                    self.assertEqual(ref, res)

    def test_get_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            self.mf_map.get("n00")
        self.assertEqual(
            "there is no format with name 'n00'", exc.exception.args[0]
        )

import shutil
import unittest
import configparser

from pyiak_instr_deprecation.rwfile import RWConfig

from tests.env_vars import TEST_DATA_DIR


CONFIG_NAME = 'test_config.ini'
CONFIG_PATH = TEST_DATA_DIR / CONFIG_NAME
REF_CONFIG_CONTENT = {
    "single": {
        "letter": "a",
        "word": "abcde",
        "str_int": "\\str(1)",
        "int": "1",
        "float": "5.4321",
        "efloat_small": "5.4321e-99",
        "efloat_huge": "5.4321e+99",
        "bool": "True",
        "none": "None",
    },
    "list": {
        "empty": "\\lst()",
        "all_types": "\\lst(a,2,3.3,4.4e-99,5.5e+99)",
    },
    "tuple": {
        "empty": "\\tpl()",
        "all_types": "\\tpl(a,2,3.3,4.4e-99,5.5e+99)",
    },
    "dict": {
        "all_types": r"\dct(a,2,3.3,4.4e-99,5.5e+99,\str(1))",
        "empty": "\\dct()",
    }
}
CONFIG_DATA = {
    "single": {
        "letter": "a",
        "word": "abcde",
        "str_int": "1",
        "int": 1,
        "float": 5.4321,
        "efloat_small": 5.4321e-99,
        "efloat_huge": 5.4321e+99,
        "bool": True,
        "none": None,
    },
    "list": {
        "empty": [],
        "all_types": ["a", 2, 3.3, 4.4e-99, 5.5e+99],
    },
    "tuple": {
        "empty": (),
        "all_types": ("a", 2, 3.3, 4.4e-99, 5.5e+99),
    },
    "dict": {
        "all_types": {"a": 2, 3.3: 4.4e-99, 5.5e+99: "1"},
        "empty": {},
    }
}


class TestRWConfig(unittest.TestCase):

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.rwc = RWConfig(CONFIG_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_DIR)

    def setUp(self) -> None:
        self.rwc.write(CONFIG_DATA)

    def test_file_creation(self):
        RWConfig(CONFIG_PATH)
        self.assertTrue(CONFIG_PATH.exists())

    def test_filepath_as_str(self):
        RWConfig(str(CONFIG_PATH))
        self.assertTrue(CONFIG_PATH.exists())

    def test_write_single(self):
        options = ["int", "float", "efloat_small", "efloat_huge"]
        vals = [10, 254.641685, 1.12476e-123, 5.4165e+123]
        vals_str = ["10", "254.641685", "1.12476e-123", "5.4165e+123"]
        for i in range(len(options)):
            with self.subTest(option=options[i]):
                self.rwc.write("single", options[i], vals[i])
                readed_config = configparser.ConfigParser()
                readed_config.read(CONFIG_PATH)
                self.assertEqual(
                    vals_str[i],
                    readed_config.get("single", options[i])
                )

    def test_write_sec_dict(self) -> None:
        path = TEST_DATA_DIR / "sec_dict.ini"
        data = {
            "sec1": {
                "opt1": "a",
                "opt2": "b",
            },
            "sec2": {
                "opt3": "c",
            }
        }
        with RWConfig(path) as rwc:
            for sec, options in data.items():
                rwc.write(sec, options)

        cfg = configparser.ConfigParser()
        cfg.read(path)

        for sec, options in data.items():
            for opt, val in options.items():
                with self.subTest(sec=sec, opt=opt):
                    self.assertEqual(val, cfg.get(sec, opt))

    def test_write_dict(self) -> None:
        path = TEST_DATA_DIR / "dict.ini"
        data = {
            "sec1": {
                "opt1": "a",
                "opt2": "b",
            },
            "sec2": {
                "opt3": "c",
            }
        }
        RWConfig(path).write(data)

        cfg = configparser.ConfigParser()
        cfg.read(path)

        for sec, options in data.items():
            for opt, val in options.items():
                with self.subTest(sec=sec, opt=opt):
                    self.assertEqual(val, cfg.get(sec, opt))

    def test_write_wrong_args(self):
        with self.assertRaises(TypeError) as exc:
            self.rwc.write("", "")
        self.assertEqual(exc.exception.args[0], "invalid arguments ('', '')")

    def test_read_conversion(self) -> None:
        for section, option_dict in CONFIG_DATA.items():
            for option, conv_value in option_dict.items():
                with self.subTest(section=section, option=option):
                    self.assertIsInstance(
                        self.rwc.read(section, option),
                        type(conv_value)
                    )
                    self.assertEqual(
                        self.rwc.read(section, option),
                        conv_value
                    )

    def test_get(self) -> None:
        sec, opt = "single", "int"
        self.assertEqual(CONFIG_DATA[sec][opt], self.rwc.get(sec, opt))

    def test_set(self) -> None:
        sec, opt, val = "single", "int", 12345
        self.rwc.set(sec, opt, val)
        self.assertEqual(val, self.rwc.get(sec, opt))

    def test_set_isolation(self) -> None:

        sec, opt, val = "single", "int", 12345
        self.rwc.set(sec, opt, val)
        self.assertEqual(val, self.rwc.get(sec, opt))
        self.assertEqual(CONFIG_DATA[sec][opt], self.rwc.read(sec, opt))

    def test_apply_changes(self) -> None:

        sec, opt, val = "single", "int", 12345
        self.assertEqual(CONFIG_DATA[sec][opt], self.rwc.get(sec, opt))
        self.rwc.set(sec, opt, val)
        self.rwc.apply_changes()
        self.assertEqual(val, self.rwc.read(sec, opt))
        self.assertEqual(val, self.rwc.get(sec, opt))

    def test_correct_writing(self) -> None:
        with open(CONFIG_PATH, "r") as file:
            res = file.readlines()

        i_line = 0
        for sec, options in REF_CONFIG_CONTENT.items():

            if i_line:
                self.assertEqual("\n", res[i_line])
                i_line += 1

            with self.subTest(line=i_line, sec=sec):
                self.assertEqual(f"[{sec}]\n", res[i_line])
            i_line += 1

            for opt, val in options.items():

                line = res[i_line]
                with self.subTest(line=i_line, sec=sec, opt=opt):
                    self.assertEqual(f"{opt} = {val}\n", line)
                i_line += 1

    def test_correct_raw_reading(self) -> None:
        for sec, options in REF_CONFIG_CONTENT.items():
            for opt, ref in options.items():
                val = self.rwc.get(sec, opt, convert=False)
                with self.subTest(sec=sec, opt=opt):
                    self.assertEqual(ref, val)

    def test_correct_reading(self) -> None:
        for sec, options in CONFIG_DATA.items():
            for opt, ref in options.items():
                val = self.rwc.get(sec, opt)
                with self.subTest(sec=sec, opt=opt):
                    self.assertEqual(ref, val)

    def test_str_magic(self):
        self.assertEqual(
            r"RWConfig('data_test\test_config.ini')", str(self.rwc)
        )

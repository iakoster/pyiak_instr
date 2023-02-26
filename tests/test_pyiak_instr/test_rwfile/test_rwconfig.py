import unittest
from pathlib import Path
from configparser import ConfigParser
from typing import Any

from src.pyiak_instr.rwfile import RWConfig

from ..env import get_local_test_data_dir, remove_test_data_dir


TEST_DATA_DIR = get_local_test_data_dir(__name__)
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

    TEST_FILE = TEST_DATA_DIR / "test_config.ini"
    maxDiff = None

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_file_creation(self):
        RWConfig(self.TEST_FILE)
        self.assertTrue(self.TEST_FILE.exists())

    def test_set_single(self):
        data = dict(
            single=dict(
                int=10,
                float=254.641685,
                efloat_small=1.12476e-123,
                efloat_huge=5.4165e+123,
            )
        )

        with RWConfig(TEST_DATA_DIR / "a.ini") as rwc:
            for sec, opts in data.items():
                for opt, val in opts.items():
                    rwc.set(sec, opt, val)

                    with self.subTest(sec=sec, opt=opt):
                        self.assertEqual(data[sec][opt], rwc.get(sec, opt))

    def test_set_sec_dict(self) -> None:
        data = {"sec1": {"opt1": "a", "opt2": "b"}, "sec2": {"opt3": "c"}}
        with RWConfig(TEST_DATA_DIR / "a.ini") as rwc:
            for sec, options in data.items():
                rwc.set(sec, options)

            for sec, options in data.items():
                for opt, val in options.items():
                    with self.subTest(sec=sec, opt=opt):
                        self.assertEqual(val, rwc.get(sec, opt))

    def test_set_dict(self) -> None:
        data = {"sec1": {"opt1": "a", "opt2": "b"}, "sec2": {"opt3": "c"}}
        with RWConfig(TEST_DATA_DIR / "a.ini") as rwc:
            rwc.set(data)

            for sec, options in data.items():
                for opt, val in options.items():
                    with self.subTest(sec=sec, opt=opt):
                        self.assertEqual(val, rwc.get(sec, opt))

    def test_set_wrong_args(self) -> None:
        with self.assertRaises(TypeError) as exc, self._get_rwc() as rwc:
            rwc.set("", "")
        self.assertEqual(exc.exception.args[0], "invalid arguments ('', '')")

    def test_isolation(self) -> None:

        def get_file_dict(path: Path) -> dict[str, dict[str, Any]]:
            return get_cfg_dict(self._get_parser(path))

        def get_cfg_dict(cfg_: ConfigParser) -> dict[str, dict[str, Any]]:
            dct = {s: {
                o: v for o, v in opts.items()
            } for s, opts in cfg_.items()}
            dct.pop("DEFAULT")
            return dct

        with self._get_rwc() as rwc:
            self.assertListEqual(
                [], self._get_parser(rwc.filepath).sections()
            )

            rwc.set(CONFIG_DATA)
            self.assertListEqual(
                [], self._get_parser(rwc.filepath).sections()
            )

            rwc.commit()
            self.assertDictEqual(
                REF_CONFIG_CONTENT, get_file_dict(rwc.filepath)
            )

            rwc.set("def", "new", "a")
            self.assertDictEqual(
                REF_CONFIG_CONTENT, get_file_dict(rwc.filepath)
            )

            ref = get_file_dict(rwc.filepath)
            ref.update({"def": {"new": "a"}})
            self.assertDictEqual(ref, get_cfg_dict(rwc.api))

            rwc.drop_changes()
            self.assertDictEqual(REF_CONFIG_CONTENT, get_cfg_dict(rwc.api))

    def _get_rwc(self) -> RWConfig:
        return RWConfig(self.TEST_FILE)

    @staticmethod
    def _get_parser(path: Path) -> ConfigParser:
        cfg = ConfigParser()
        cfg.read(path)
        return cfg

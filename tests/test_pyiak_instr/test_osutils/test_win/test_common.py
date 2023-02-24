import unittest
from subprocess import call

from ...env import remove_test_data_dir, get_local_test_data_dir

from src.pyiak_instr.osutils import is_hidden_path, hide_path, unhide_path


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestCommonFunctions(unittest.TestCase):

    TEST_FILE = TEST_DATA_DIR / "test.txt"

    @classmethod
    def setUpClass(cls) -> None:
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cls.TEST_FILE, "w"):
            ...

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_hide_path(self) -> None:
        call(["attrib", "-h", self.TEST_FILE])
        self.assertFalse(is_hidden_path(self.TEST_FILE))
        hide_path(self.TEST_FILE)
        self.assertTrue(is_hidden_path(self.TEST_FILE))

    def test_hide_path_exc(self) -> None:
        with self.assertRaises(FileNotFoundError) as exc:
            hide_path(TEST_DATA_DIR / "not_existed")
        self.assertEqual("path not found", exc.exception.args[0])

    def test_is_hidden_path(self) -> None:
        call(["attrib", "-h", self.TEST_FILE])
        self.assertFalse(is_hidden_path(self.TEST_FILE))
        call(["attrib", "+h", self.TEST_FILE])
        self.assertTrue(is_hidden_path(self.TEST_FILE))

    def test_is_hidden_path_exc(self) -> None:
        with self.assertRaises(FileNotFoundError) as exc:
            is_hidden_path(TEST_DATA_DIR / "not_existed")
        self.assertEqual("path not found", exc.exception.args[0])

    def test_unhide_path(self) -> None:
        call(["attrib", "+h", self.TEST_FILE])
        self.assertTrue(is_hidden_path(self.TEST_FILE))
        unhide_path(self.TEST_FILE)
        self.assertFalse(is_hidden_path(self.TEST_FILE))

    def test_unhide_path_exc(self) -> None:
        with self.assertRaises(FileNotFoundError) as exc:
            unhide_path(TEST_DATA_DIR / "not_existed")
        self.assertEqual("path not found", exc.exception.args[0])


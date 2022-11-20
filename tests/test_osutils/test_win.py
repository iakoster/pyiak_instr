import shutil
import unittest

from pyinstr_iakoster.osutils import (
    hide_path,
    unhide_path,
    is_hidden_path,
)

from ..env_vars import TEST_DATA_DIR


class TestCommon(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TEST_DATA_DIR.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def setUp(self) -> None:
        (TEST_DATA_DIR / "dir").mkdir(exist_ok=True)
        with open(TEST_DATA_DIR / "file.txt", "w"):
            ...

    def test_hide_unhide(self):

        for path in (TEST_DATA_DIR / "dir", TEST_DATA_DIR / "file.txt"):
            with self.subTest(path=path):
                hide_path(path)
                self.assertTrue(is_hidden_path(path))
                unhide_path(path)
                self.assertFalse(is_hidden_path(path))

    def test_hide_unhide_exc(self):
        for func in (
            hide_path, unhide_path, is_hidden_path
        ):
            with self.subTest(method=func.__name__):
                with self.assertRaises(FileExistsError) as exc:
                    func(TEST_DATA_DIR / "test")
                self.assertEqual("path not exists", exc.exception.args[0])

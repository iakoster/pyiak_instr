import shutil
import unittest
from pathlib import Path

from pyinstr_iakoster.rwfile import RWFile, RWFileError, FileSuffixError

from tests.env_vars import TEST_DATA_DIR


class TestRWFile(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_DIR)

    def test_init_invalid_suffix(self):
        RWFile.FILE_SUFFIXES = {".any"}
        with self.assertRaises(FileSuffixError) as exc:
            RWFile("test_data/test.not")
        self.assertEqual(
            r"suffix of 'test_data\test.not' not in {'.any'}",
            exc.exception.args[0]
        )
        self.assertFalse(Path("test_data").exists())
        RWFile.FILE_SUFFIXES = {}

    def test_init_not_file(self):
        with self.assertRaises(RWFileError) as exc:
            RWFile(".")
        self.assertEqual("path not lead to file", exc.exception.args[0])

    def test_init_with_mkdir(self):
        RWFile(TEST_DATA_DIR / "test.any")
        self.assertTrue(TEST_DATA_DIR.exists())

    def test_filepath(self):
        self.assertEqual(Path("test.any"), RWFile("test.any").filepath)

    def test_with(self):
        with self.assertRaises(NotImplementedError):
            with RWFile(Path("test.any")) as rwf:
                self.assertEqual(Path("test.any"), rwf.filepath)

    def test_str_magic(self):
        self.assertEqual(
            "RWFile('test.any')",
            str(RWFile("test.any"))
        )

    def test_repr_magic(self):
        self.assertEqual(
            "<RWFile('test.any')>",
            repr(RWFile("test.any"))
        )

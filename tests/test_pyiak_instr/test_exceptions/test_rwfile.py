import unittest
from pathlib import Path

from src.pyiak_instr.exceptions import RWFileError, FileSuffixError


class TestRWFileError(unittest.TestCase):

    def test_init(self) -> None:
        path = Path("C://directory")
        with self.assertRaises(RWFileError) as exc:
            raise RWFileError("test case", path)
        res = exc.exception

        self.assertTupleEqual(("test case", path), res.args)
        self.assertEqual(path, res.filepath)


class TestFileSuffixError(unittest.TestCase):

    def test_init(self) -> None:
        path = Path("C://directory/file.txt")
        with self.assertRaises(FileSuffixError) as exc:
            raise FileSuffixError({".ini"}, path)
        res = exc.exception

        self.assertTupleEqual(
            (
                r"suffix of 'C:\directory\file.txt' not in {'.ini'}",
                path,
                ".txt",
            ),
            res.args,
        )
        self.assertEqual(path, res.filepath)
        self.assertSetEqual({".ini"}, res.suffixes)

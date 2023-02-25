import unittest
from pathlib import Path

from src.pyiak_instr.rwfile import RWFile, FileSuffixError

from ..env import get_local_test_data_dir, remove_test_data_dir
from ...utils import validate_object


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class RWFileTestInstance(RWFile[str]):

    def __init__(self, filepath: Path | str):
        super().__init__(filepath, "api test instance")

    def close(self) -> None:
        pass


class TestRWFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_init(self) -> None:
        validate_object(
            self,
            RWFileTestInstance("test.true"),
            api="api test instance",
            filepath=Path("test.true"),
        )

    def test_init_with_mkdir(self) -> None:
        dir_ = get_local_test_data_dir("temp")
        RWFileTestInstance(dir_ / "test.test")
        self.assertTrue(dir_.exists())

    def test_init_exc(self) -> None:
        RWFile.ALLOWED_SUFFIXES = {".true"}
        with self.assertRaises(FileSuffixError) as exc:
            RWFileTestInstance("test.false")
        self.assertTupleEqual(
            (
                r"suffix of 'test.false' not in {'.true'}",
                Path("test.false"),
                ".false",
            ),
            exc.exception.args
        )
        RWFile.ALLOWED_SUFFIXES = {}

        with self.assertRaises(FileNotFoundError) as exc:
            RWFileTestInstance("")
        self.assertEqual("path not lead to file", exc.exception.args[0])

    def test_context_manager(self) -> None:
        with RWFileTestInstance(Path("test.any")) as rwf:
            self.assertIsInstance(rwf, RWFile)

    def test_magic_str_repr(self) -> None:
        str_ = "RWFileTestInstance('test.any')"
        repr_ = "<%s>" % str_
        obj = RWFileTestInstance("test.any")
        self.assertEqual(str_, str(obj))
        self.assertEqual(repr_, repr(obj))

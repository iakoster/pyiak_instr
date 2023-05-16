import unittest
from pathlib import Path

from src.pyiak_instr.types import RWData
from src.pyiak_instr.exceptions import FileSuffixError, NotSupportedMethod

from ..env import get_local_test_data_dir, remove_test_data_dir
from ...utils import validate_object


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TIRWData(RWData[str]):

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _get_api(self, filepath: Path) -> str:
        return f"API: {filepath}"


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
            TIRWData(Path("test.true")),
            api="API: test.true",
            filepath=Path("test.true"),
        )

    def test_init_with_mkdir(self) -> None:
        dir_ = get_local_test_data_dir("temp")
        TIRWData(dir_ / "test.test")
        self.assertTrue(dir_.exists())

    def test_init_exc(self) -> None:
        TIRWData.ALLOWED_SUFFIXES = {".true"}
        with self.assertRaises(FileSuffixError) as exc:
            TIRWData(Path("test.false"))
        self.assertTupleEqual(
            (
                r"suffix of 'test.false' not in {'.true'}",
                Path("test.false"),
                ".false",
            ),
            exc.exception.args
        )
        TIRWData.ALLOWED_SUFFIXES = {}

        with self.assertRaises(FileNotFoundError) as exc:
            TIRWData(Path(""))
        self.assertEqual("path not lead to file", exc.exception.args[0])

    def test_not_supported_methods(self) -> None:
        obj = TIRWData(Path("temp"))
        for method in ("request", "get", "set", "read", "write"):
            with self.subTest(test=method):
                with self.assertRaises(NotSupportedMethod) as exc:
                    getattr(obj, method)()
                self.assertEqual(
                    f"RWData does not support .{method}", exc.exception.args[0]
                )

    def test_context_manager(self) -> None:
        with TIRWData(Path("test.any")) as rwf:
            self.assertIsInstance(rwf, TIRWData)

    def test_magic_str_repr(self) -> None:
        str_ = "TIRWData('test.any')"
        repr_ = "<%s>" % str_
        obj = TIRWData(Path("test.any"))
        self.assertEqual(str_, str(obj))
        self.assertEqual(repr_, repr(obj))

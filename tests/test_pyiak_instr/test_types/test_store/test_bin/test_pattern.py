import unittest
import shutil

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotConfiguredYet

from .....utils import validate_object
from ....env import get_local_test_data_dir
from .ti import TIFieldStructPattern, TIStorageStructPattern, TIStoragePattern


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestBytesFieldStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIFieldStructPattern(typename="basic", bytes_expected=0),
            typename="basic",
            is_dynamic=True,
            size=0,
        )

    def test_get(self) -> None:
        validate_object(
            self,
            TIFieldStructPattern(
                typename="basic", bytes_expected=4, fmt=Code.U16
            ).get(),
            has_default=False,
            name="",
            stop=4,
            start=0,
            words_expected=2,
            order=Code.BIG_ENDIAN,
            word_bytesize=2,
            bytes_expected=4,
            is_dynamic=False,
            default=b"",
            slice_=slice(0, 4),
            fmt=Code.U16,
            wo_attrs=["encoder"],
        )


class TestBytesStorageStructPatternABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            TIStorageStructPattern(typename="basic", name="test"),
            typename="basic",
        )

    def test_get(self) -> None:
        storage = TIStorageStructPattern(
            typename="basic", name="test"
        ).configure(
                f0=TIFieldStructPattern(typename="basic", bytes_expected=3)
            ).get()

        validate_object(
            self,
            storage,
            dynamic_field_name="",
            is_dynamic=False,
            minimum_size=3,
            name="test",
            fields={},
        )
        validate_object(
            self,
            storage["f0"],
            has_default=False,
            name="f0",
            stop=3,
            start=0,
            words_expected=3,
            order=Code.BIG_ENDIAN,
            word_bytesize=1,
            bytes_expected=3,
            is_dynamic=False,
            default=b"",
            slice_=slice(0, 3),
            fmt=Code.U8,
            wo_attrs=["encoder"],
        )


class TestBytesStoragePatternABC(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_init(self) -> None:
        validate_object(
            self,
            TIStoragePattern(typename="basic", name="test"),
            typename="basic",
        )

    def test_get(self) -> None:
        storage_struct = TIStorageStructPattern(
            typename="basic"
        )
        storage_struct.configure(
            f0=TIFieldStructPattern(typename="basic", bytes_expected=2)
        )
        storage = TIStoragePattern(typename="basic").configure(
                test=storage_struct
            )
        res = storage.get()

        validate_object(
            self,
            res,
            has_pattern=True,
            wo_attrs=["struct", "pattern"],
        )
        validate_object(
            self,
            res.struct,
            dynamic_field_name="",
            is_dynamic=False,
            minimum_size=2,
            name="test",
            fields={},
        )

    def test_configure_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIStoragePattern(typename="basic").configure(
                s0=TIStorageStructPattern(typename="basic"),
                s1=TIStorageStructPattern(typename="basic"),
            )
        self.assertEqual(
            "only one storage pattern allowed, got 2", exc.exception.args[0]
        )

    def test_write(self) -> None:
        path = TEST_DATA_DIR / "test_write.ini"
        self._instance().write(path)

        ref = [
            "[test]",
            r"_ = \dct(typename,basic,val,\str(33))",
            r"test = \dct(typename,basic,val,\tpl(11))",
            r"first = \dct(typename,basic,bytes_expected,0,int,3,"
            r"list,\lst(2,3,4))",
            r"second = \dct(typename,basic,bytes_expected,0,boolean,True)",
            r"third = \dct(typename,basic,bytes_expected,0,"
            r"dict,\dct(0,1,2,3))",
        ]
        i_line = 0
        with open(path, "r") as file:
            for rf, rs in zip(ref, file.read().split("\n")):
                i_line += 1
                with self.subTest(test="new", line=i_line):
                    self.assertEqual(rf, rs)
        self.assertEqual(len(ref), i_line)

        TIStoragePattern(typename="basic").configure(
            test=TIStorageStructPattern(
                typename="basic", in33=0
            ).configure(
                f0=TIFieldStructPattern(typename="basic", bytes_expected=33)
            ),
        ).write(path)

        ref = [
            "[test]",
            r"_ = \dct(typename,basic)",
            r"test = \dct(typename,basic,in33,0)",
            r"f0 = \dct(typename,basic,bytes_expected,33)",
        ]
        i_line = 0
        with open(path, "r") as file:
            for rf, rs in zip(ref, file.read().split("\n")):
                i_line += 1
                with self.subTest(test="rewrite", line=i_line):
                    self.assertEqual(rf, rs)
        self.assertEqual(len(ref), i_line)

    def test_write_exc_not_configured(self) -> None:
        with self.assertRaises(NotConfiguredYet) as exc:
            TIStoragePattern(typename="basic", name="test", val=(11,)).write(
                TEST_DATA_DIR / "test.ini"
            )
        self.assertEqual(
            "TIStoragePattern not configured yet", exc.exception.args[0]
        )

    def test_write_read(self) -> None:
        path = TEST_DATA_DIR / "test_write_read.ini"
        ref = self._instance()
        ref.write(path)
        res = TIStoragePattern.read(path, "test")

        self.assertIsNot(ref, res)
        self.assertEqual(ref, res)
        self.assertEqual(ref._sub_p, res._sub_p)
        self.assertEqual(ref._sub_p["test"]._sub_p, res._sub_p["test"]._sub_p)

    def test_read_exc(self) -> None:
        with self.assertRaises(TypeError) as exc:
            TIStoragePattern.read(TEST_DATA_DIR, "key_1", "key_2")
        self.assertEqual(
            "TIStoragePattern takes only 1 argument (2 given)",
            exc.exception.args[0],
        )

    @staticmethod
    def _instance() -> TIStoragePattern:
        struct_pattern = TIStorageStructPattern(
            typename="basic", val=(11,)
        )
        struct_pattern.configure(
            first=TIFieldStructPattern(
                typename="basic", bytes_expected=0, int=3, list=[2, 3, 4]
            ),
            second=TIFieldStructPattern(
                typename="basic", bytes_expected=0, boolean=True
            ),
            third=TIFieldStructPattern(
                typename="basic", bytes_expected=0, dict={0: 1, 2: 3}
            )
        )
        pattern = TIStoragePattern(typename="basic", val="33")
        pattern.configure(test=struct_pattern)
        return pattern

import shutil
import unittest

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.rwfile import RWExcel
from pyinstr_iakoster.rwfile import FilepathPatternError

EXCEL_NAME = 'test_excel.xlsx'
EXCEL_PATH = DATA_TEST_DIR / EXCEL_NAME


class TestRWExcel(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if DATA_TEST_DIR.exists():
            shutil.rmtree(DATA_TEST_DIR)

    def setUp(self) -> None:
        if not EXCEL_PATH.exists():
            RWExcel.new_empty(EXCEL_PATH, first_sheet='test_sheet')

    def test_init_wrong_path(self):
        with self.assertRaises(FilepathPatternError) as exc:
            RWExcel(r'.\abrakadabre')
        self.assertEqual(
            'The path does not lead to \'\\\\S+.xlsx$\' file',
            exc.exception.message
        )

    def test_create_empty_wrong_path(self):
        with self.assertRaises(FilepathPatternError) as exc:
            RWExcel.new_empty(r'.\abrakaderbre')
        self.assertEqual(
            'The path does not lead to \'\\\\S+.xlsx$\' file',
            exc.exception.message
        )

    def test_create_empty(self):
        self.tearDownClass()
        RWExcel.new_empty(EXCEL_PATH, first_sheet='test_sheet')

    def test_create_empty_again(self):
        with self.assertRaises(FileExistsError) as exc:
            RWExcel.new_empty(EXCEL_PATH)
        self.assertEqual(
            exc.exception.args[0],
            'Excel file is already exists'
        )

    def test_init(self):
        rwe = RWExcel(EXCEL_PATH)
        self.assertEqual('test_sheet', rwe.excel.active.title)

    def test_cell_gs_cell(self):
        rwe = RWExcel(EXCEL_PATH, autosave=True)
        rwe.cell('A1', value=10)
        self.assertEqual(rwe.cell('A1').value, 10)

    def test_cell_gs_rowcol(self):
        rwe = RWExcel(EXCEL_PATH, autosave=True)
        rwe.cell(1, 0, value=20)
        self.assertEqual(rwe.cell(1, 0).value, 20)

    def test_cell_incorrect(self):
        rwe = RWExcel(EXCEL_PATH)
        with self.assertRaises(ValueError) as exc:
            rwe.cell(1, value=20)
        self.assertEqual(
            exc.exception.args[0],
            'Incorrect input: (1,)'
        )

    def test_magic_getitem(self):
        rwe = RWExcel(EXCEL_PATH)
        rwe.cell('A1', value=10)
        self.assertEqual(rwe['A1'].value, 10)
        self.assertEqual(rwe[0, 0].value, 10)

    def test_magic_getitem_wrong_return(self):
        rwe = RWExcel(EXCEL_PATH)
        with self.assertRaises(AssertionError) as exc:
            a = rwe['A1:A10']
        self.assertEqual(
            exc.exception.args[0],
            'incorrect .cell return: <class \'tuple\'>'
        )

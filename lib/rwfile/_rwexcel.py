import re
from pathlib import Path
from typing import overload, Any

import openpyxl as opxl
from openpyxl.cell.cell import Cell


class RWExcel(object):

    _FILENAME_PATTERN = re.compile('\w+.xlsx$')

    def __init__(self, filepath: Path | str, autosave: bool = False):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if self._FILENAME_PATTERN.match(filepath.name) is None:
            raise ValueError(
                'The specified path does not lead to '
                f'the *.xlsx file: {filepath}')
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

        self._filepath = filepath
        self._autosave = autosave
        self._xcl = opxl.open(self._filepath)

    def change_active(self, sheet_name: str) -> None:
        self._xcl.active = self._xcl[sheet_name]

    def save(self) -> None:
        self._xcl.save(self._filepath)

    @classmethod
    def create_empty(
            cls,
            filepath: Path | str,
            sheet_name: str = 'Sheet'
    ) -> None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if cls._FILENAME_PATTERN.match(filepath.name) is None:
            raise ValueError(
                'The specified path does not lead to '
                f'the *.xlsx file: {filepath}')
        if filepath.exists():
            raise FileExistsError('Excel file is already exists')
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

        wb = opxl.Workbook()
        wb.active.title = sheet_name
        wb.save(filepath)

    @overload
    def cell(self, cell_name: str, value: Any = None) -> Cell:
        ...

    @overload
    def cell(self, row: int, col: int, value: Any = None) -> Cell:
        ...

    def cell(self, *coords, **kwargs) -> Cell:
        match coords:
            case (str() as cell_name,):
                cell = self._xcl.active[cell_name]
            case (int() as row, int() as col):
                cell = self._xcl.active.cell(row + 1, col + 1)
            case _:
                raise ValueError(f'Incorrect input: {coords}')
        if 'value' in kwargs:
            cell.value = kwargs['value']
            if self._autosave:
                self.save()
        return cell

    @property
    def filepath(self):
        return self._filepath

    @property
    def excel(self):
        return self._xcl

    def __getitem__(self, *coords: str | tuple[int, int]) -> Cell:
        match coords:
            case ((int(), int()),):
                coords = coords[0]
        res = self.cell(*coords)
        if isinstance(res, Cell):
            return res
        else:
            assert False, f'incorrect .cell return: {type(res)}'
